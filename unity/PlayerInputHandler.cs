using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>Lima state kontrol mata. Hanya ini yang ada.</summary>
public enum EyeState { Neutral, Left, Right, Up, Down }

/// <summary>
/// Pengganti PlayerInputHandler lama: input diambil dari lirikan mata (paket UDP dari skrip Python),
/// bukan keyboard/gamepad. API publik dibuat sama dengan yang dipakai PlayerMovement:
///   MovementVector, IsMoving, JumpPressed, JumpReleased
/// ditambah AttackPressed / AttackReleased untuk serangan.
///
/// Pemetaan state -> aksi:
///   Left    -> jalan ke kiri     (MovementVector.x = -1)
///   Right   -> jalan ke kanan    (MovementVector.x = +1)
///   Up      -> lompat            (JumpPressed saat masuk Up, JumpReleased saat keluar: tahan = lompat tinggi)
///   Down    -> serang            (AttackPressed saat masuk Down, AttackReleased saat keluar)
///   Neutral -> diam
/// Wajah hilang / paket basi / keyakinan rendah dianggap Neutral.
///
/// Pasang di GameObject pemain (yang sama dengan PlayerMovement). File ini MENGGANTIKAN
/// PlayerInputHandler.cs lama: simpan dulu yang lama di luar folder Assets.
/// </summary>
[DisallowMultipleComponent]
public class PlayerInputHandler : MonoBehaviour
{
    // ---------------- API untuk PlayerMovement dan skrip lain ----------------
    public event Action JumpPressed;
    public event Action JumpReleased;
    public event Action AttackPressed;
    public event Action AttackReleased;

    public Vector2 MovementVector { get; private set; }
    public bool IsMoving { get { return Mathf.Abs(MovementVector.x) > 0.01f; } }
    public EyeState State { get { return _state; } }

    /// <summary>Matikan sementara (cutscene, menu). State dikembalikan ke Neutral.</summary>
    public bool InputEnabled
    {
        get { return _inputEnabled; }
        set { _inputEnabled = value; if (!value) ForceNeutral(); }
    }

    // ---------------- Pengaturan (Inspector) ----------------
    [Header("Jaringan")]
    [SerializeField] private int port = 5005;
    [SerializeField] private float maxAgeMs = 250f;          // tidak ada paket baru selama ini -> Neutral

    [Header("Penstabil (FSM)")]
    [SerializeField] private float minConf = 0.6f;           // di bawah ini dianggap tidak yakin (Neutral)
    [SerializeField] private int enterFrames = 3;            // paket berturut-turut agar pindah ke Left/Right/Up/Down
    [SerializeField] private int releaseFrames = 2;          // paket berturut-turut agar kembali ke Neutral
    [SerializeField] private float jumpCooldownS = 0.15f;    // jeda minimum antar JumpPressed
    [SerializeField] private float attackCooldownS = 0.30f;  // jeda minimum antar AttackPressed

    [Header("Lompat sambil bergerak")]
    [Tooltip("Satu mata hanya punya satu arah lirikan. Jika aktif, arah jalan sebelum melirik ke atas dipertahankan selama Up.")]
    [SerializeField] private bool carryDirectionIntoJump = true;
    [SerializeField] private float carryWindowS = 0.35f;     // Left/Right harus baru saja aktif sebelum Up

    [Header("Info debug (hanya baca)")]
    [SerializeField] private EyeState debugState;
    [SerializeField] private string debugLabel;
    [SerializeField] private float debugConf;
    [SerializeField] private float debugPacketAgeMs;
    [SerializeField] private bool debugLogStateChanges;

    // ---------------- Internal ----------------
    [Serializable]
    public class EyePacket
    {
        public long frame_id;
        public bool face;
        public string label;
        public float conf;
    }

    private EyeState _state = EyeState.Neutral;
    private EyeState _candidate = EyeState.Neutral;
    private int _candidateCount;
    private bool _inputEnabled = true;

    private bool _jumpHeld, _attackHeld;
    private float _lastJumpTime = -999f, _lastAttackTime = -999f;
    private float _lastHorizTime = -999f;
    private float _lastHorizDir;
    private float _carryDir;

    // thread penerima
    private UdpClient _client;
    private Thread _thread;
    private volatile bool _running;
    private readonly object _lock = new object();
    private string _latestJson;
    private long _latestTicks;
    private string _parsedJson;
    private long _lastFrameId = -1;
    private EyePacket _packet;

    private static readonly Stopwatch Clock = Stopwatch.StartNew();

    // ---------------- Siklus hidup ----------------
    private void Awake()
    {
        Application.runInBackground = true;   // tetap menerima input saat jendela Python yang fokus
    }

    private void OnEnable()
    {
        StartReceiver();
    }

    private void OnDisable()
    {
        ForceNeutral();
        StopReceiver();
    }

    private void StartReceiver()
    {
        if (_running) return;
        try
        {
            _client = new UdpClient(new IPEndPoint(IPAddress.Loopback, port));
            _running = true;
            _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "EyeUdpReceiver" };
            _thread.Start();
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("PlayerInputHandler: gagal membuka UDP port " + port +
                " (port dipakai skrip Unity lain? matikan GazeUdpReceiver lama). " + e.Message);
        }
    }

    private void StopReceiver()
    {
        _running = false;
        try { if (_client != null) _client.Close(); } catch (Exception) { }
        _client = null;
        _thread = null;
    }

    private void ReceiveLoop()
    {
        IPEndPoint any = new IPEndPoint(IPAddress.Any, 0);
        while (_running)
        {
            try
            {
                byte[] data = _client.Receive(ref any);
                string json = Encoding.UTF8.GetString(data);
                lock (_lock) { _latestJson = json; _latestTicks = Clock.ElapsedTicks; }
            }
            catch (SocketException) { if (!_running) break; Thread.Sleep(10); }
            catch (ObjectDisposedException) { break; }
            catch (Exception) { if (!_running) break; Thread.Sleep(10); }
        }
    }

    // ---------------- Per frame ----------------
    private void Update()
    {
        float now = Time.unscaledTime;

        // 1) ambil paket terbaru (parse di thread utama)
        string json; long ticks;
        lock (_lock) { json = _latestJson; ticks = _latestTicks; }

        float ageMs = json == null ? float.MaxValue
            : (float)((Clock.ElapsedTicks - ticks) * 1000.0 / Stopwatch.Frequency);
        debugPacketAgeMs = ageMs > 1e8f ? -1f : ageMs;

        bool fresh = json != null && ageMs <= maxAgeMs;

        if (!_inputEnabled)
        {
            Apply(EyeState.Neutral, now);
            return;
        }

        if (!fresh)
        {
            // tidak ada data baru: langsung aman (Neutral), tanpa menunggu debounce
            _candidate = EyeState.Neutral;
            _candidateCount = 0;
            Apply(EyeState.Neutral, now);
            return;
        }

        if (!ReferenceEquals(json, _parsedJson))
        {
            _parsedJson = json;
            try { _packet = JsonUtility.FromJson<EyePacket>(json); }
            catch (Exception) { _packet = null; }

            // proses hanya paket BARU (frame_id naik); kamera yang restart mengulang dari kecil -> terima juga
            if (_packet != null && _packet.frame_id != _lastFrameId)
            {
                _lastFrameId = _packet.frame_id;
                debugLabel = _packet.label;
                debugConf = _packet.conf;
                Step(RawToState(_packet), now);
            }
        }

        // dipanggil tiap frame agar tombol yang tertunda cooldown tetap ditekan begitu cooldown selesai
        Apply(_state, now);
    }

    private EyeState RawToState(EyePacket p)
    {
        if (!p.face || p.conf < minConf || string.IsNullOrEmpty(p.label)) return EyeState.Neutral;
        switch (p.label)
        {
            case "left":  return EyeState.Left;
            case "right": return EyeState.Right;
            case "up":    return EyeState.Up;
            case "down":  return EyeState.Down;
            default:      return EyeState.Neutral;   // "center", "none", dan label tak dikenal
        }
    }

    /// <summary>Debounce: state baru harus muncul beberapa paket berturut-turut.</summary>
    private void Step(EyeState observed, float now)
    {
        if (observed == _state)
        {
            _candidate = _state;
            _candidateCount = 0;
            return;
        }

        if (observed == _candidate) _candidateCount++;
        else { _candidate = observed; _candidateCount = 1; }

        int need = observed == EyeState.Neutral ? releaseFrames : enterFrames;
        if (_candidateCount >= need)
        {
            _candidateCount = 0;
            Apply(observed, now);
        }
    }

    /// <summary>Menerapkan state: memperbarui MovementVector dan memicu event.</summary>
    private void Apply(EyeState next, float now)
    {
        EyeState prev = _state;

        if (next != prev)
        {
            if (debugLogStateChanges) UnityEngine.Debug.Log("EyeState: " + prev + " -> " + next);

            // ingat arah jalan terakhir untuk "lompat sambil bergerak"
            if (prev == EyeState.Left || prev == EyeState.Right)
            {
                _lastHorizDir = prev == EyeState.Left ? -1f : 1f;
                _lastHorizTime = now;
            }
            if (next == EyeState.Up)
            {
                _carryDir = 0f;
                if (carryDirectionIntoJump && (now - _lastHorizTime) <= carryWindowS)
                    _carryDir = _lastHorizDir;
            }
            _state = next;
        }
        debugState = _state;

        // --- gerak horizontal ---
        float x = 0f;
        if (_state == EyeState.Left) x = -1f;
        else if (_state == EyeState.Right) x = 1f;
        else if (_state == EyeState.Up && carryDirectionIntoJump) x = _carryDir;
        MovementVector = new Vector2(x, 0f);

        // --- lompat: masuk Up = tekan, keluar Up = lepas (pasangan selalu seimbang) ---
        if (_state == EyeState.Up)
        {
            if (!_jumpHeld && (now - _lastJumpTime) >= jumpCooldownS)
            {
                _jumpHeld = true;
                _lastJumpTime = now;
                if (JumpPressed != null) JumpPressed();
            }
        }
        else if (_jumpHeld)
        {
            _jumpHeld = false;
            if (JumpReleased != null) JumpReleased();
        }

        // --- serang: masuk Down = tekan, keluar Down = lepas ---
        if (_state == EyeState.Down)
        {
            if (!_attackHeld && (now - _lastAttackTime) >= attackCooldownS)
            {
                _attackHeld = true;
                _lastAttackTime = now;
                if (AttackPressed != null) AttackPressed();
            }
        }
        else if (_attackHeld)
        {
            _attackHeld = false;
            if (AttackReleased != null) AttackReleased();
        }
    }

    /// <summary>Paksa Neutral dan lepas semua tombol virtual yang sedang "ditekan".</summary>
    public void ForceNeutral()
    {
        _candidate = EyeState.Neutral;
        _candidateCount = 0;
        Apply(EyeState.Neutral, Time.unscaledTime);
    }
}
