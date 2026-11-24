// Finite State Machine: mengubah label mentah per frame (berisik) menjadi perintah yang stabil.
// Kelas biasa (bukan MonoBehaviour) agar mudah diuji.
public enum GazeState { Neutral, Left, Right, Down, ActionFired, Lost }

public class GazeFSM
{
    // Parameter (dapat diubah dari luar)
    public float MinConf = 0.6f;        // prediksi di bawah ini dianggap tidak ada
    public int StableFrames = 3;        // label harus sama sekian paket berturut-turut
    public float CooldownS = 0.4f;      // jeda setelah aksi
    public float LostTimeoutS = 0.5f;   // wajah/keyakinan hilang selama ini -> Lost

    // Keluaran untuk game
    public GazeState State { get; private set; } = GazeState.Neutral;
    public float Horizontal { get; private set; }
    public bool ActionPressed { get; private set; }   // true hanya pada satu Tick
    public bool Down { get { return State == GazeState.Down; } }

    string _cand = "center";
    string _stable = "center";
    int _candCount;
    float _cooldown, _lostTimer;

    /// Panggil sekali per paket baru (dt = detik sejak Tick sebelumnya).
    public void Tick(string label, float conf, bool face, float dt)
    {
        ActionPressed = false;
        if (_cooldown > 0f) _cooldown -= dt;

        // 1) wajah hilang atau keyakinan rendah
        if (!face || conf < MinConf)
        {
            _lostTimer += dt;
            if (_lostTimer >= LostTimeoutS)
            {
                State = GazeState.Lost;
                Horizontal = 0f;
                _stable = "center"; _cand = "center"; _candCount = 0;
            }
            return;
        }
        _lostTimer = 0f;

        // 2) debounce: label harus stabil beberapa paket
        if (label == _cand) _candCount++; else { _cand = label; _candCount = 1; }
        if (_candCount >= StableFrames) _stable = _cand;

        // 3) petakan label stabil ke state
        switch (_stable)
        {
            case "left":  State = GazeState.Left;  Horizontal = -1f; break;
            case "right": State = GazeState.Right; Horizontal = 1f;  break;
            case "down":  State = GazeState.Down;  Horizontal = 0f;  break;
            case "up":
                Horizontal = 0f;
                if (State != GazeState.ActionFired)
                {
                    if (_cooldown <= 0f) { State = GazeState.ActionFired; ActionPressed = true; _cooldown = CooldownS; }
                    else State = GazeState.Neutral;
                }
                break;
            default:      State = GazeState.Neutral; Horizontal = 0f; break;
        }
    }
}
