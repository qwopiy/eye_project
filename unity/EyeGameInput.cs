using UnityEngine;

/// Sumber input dari mata. Taruh di satu GameObject bersama GazeUdpReceiver, lalu seret receiver ke slot.
public class EyeGameInput : MonoBehaviour, IGameInput
{
    [SerializeField] GazeUdpReceiver receiver;
    [SerializeField] float maxAgeMs = 250f;   // paket lebih tua dari ini dianggap basi

    [Header("Parameter FSM")]
    [SerializeField] float minConf = 0.6f;
    [SerializeField] int stableFrames = 3;
    [SerializeField] float cooldownS = 0.4f;
    [SerializeField] float lostTimeoutS = 0.5f;

    [Header("Info debug (hanya baca)")]
    public GazeState state;
    public string lastLabel;
    public float lastConf;
    public float latencyMs;

    GazeFSM _fsm;
    long _lastFrameId = -1;
    float _dtAccum;
    bool _actionLatched;

    void Awake() { _fsm = new GazeFSM(); ApplyConfig(); }
    void OnValidate() { if (_fsm != null) ApplyConfig(); }

    void ApplyConfig()
    {
        _fsm.MinConf = minConf;
        _fsm.StableFrames = stableFrames;
        _fsm.CooldownS = cooldownS;
        _fsm.LostTimeoutS = lostTimeoutS;
    }

    void Update()
    {
        _dtAccum += Time.unscaledDeltaTime;

        GazeMessage msg = null; double ageMs = double.MaxValue, lat = 0;
        bool have = receiver != null && receiver.TryGetLatest(out msg, out ageMs, out lat);

        if (have && ageMs <= maxAgeMs)
        {
            if (msg.frame_id != _lastFrameId)   // hanya proses paket BARU
            {
                _lastFrameId = msg.frame_id;
                _fsm.Tick(msg.label, msg.conf, msg.face, _dtAccum);
                _dtAccum = 0f;
                lastLabel = msg.label; lastConf = msg.conf; latencyMs = (float)lat;
            }
        }
        else
        {
            _fsm.Tick("center", 0f, false, _dtAccum);   // tidak ada data segar -> anggap wajah hilang
            _dtAccum = 0f;
        }

        if (_fsm.ActionPressed) _actionLatched = true;
        state = _fsm.State;
    }

    public float GetHorizontal() { return _fsm.Horizontal; }
    public bool GetDown() { return _fsm.Down; }
    public bool GetActionDown() { bool a = _actionLatched; _actionLatched = false; return a; }
}
