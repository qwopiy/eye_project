using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class GazeMessage
{
    public int v;
    public long frame_id;
    public double t_capture_ms;   // waktu epoch (ms) saat frame diambil oleh Python
    public bool face;
    public string label;          // "center" | "left" | "right" | "up" | "down" | "none"
    public float conf;
    public float[] probs;
    public string[] classes;
    public string model;
    public float infer_ms;
}

/// Menerima paket UDP JSON dari skrip Python di thread terpisah.
/// Parsing JSON dilakukan di thread utama (JsonUtility aman dipakai di sana).
public class GazeUdpReceiver : MonoBehaviour
{
    [SerializeField] int port = 5005;

    UdpClient _client;
    Thread _thread;
    volatile bool _running;
    readonly object _lock = new object();
    string _latestJson;
    double _latestRecvMs;
    string _parsedJson;
    GazeMessage _parsed;

    static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
    static double NowMs() { return (DateTime.UtcNow - Epoch).TotalMilliseconds; }

    void OnEnable()
    {
        Application.runInBackground = true;   // tetap jalan saat jendela Python yang fokus
        _running = true;
        _client = new UdpClient(port);
        _thread = new Thread(Loop) { IsBackground = true };
        _thread.Start();
    }

    void OnDisable()
    {
        _running = false;
        if (_client != null) _client.Close();
        if (_thread != null) _thread.Join(300);
        _thread = null;
        _client = null;
    }

    void Loop()
    {
        var ep = new IPEndPoint(IPAddress.Any, 0);
        while (_running)
        {
            try
            {
                byte[] data = _client.Receive(ref ep);
                string json = Encoding.UTF8.GetString(data);
                double now = NowMs();
                lock (_lock) { _latestJson = json; _latestRecvMs = now; }
            }
            catch (SocketException) { if (!_running) break; }
            catch (ObjectDisposedException) { break; }
        }
    }

    /// Ambil paket terbaru. ageMs = umur sejak diterima; latencyMs = sejak frame diambil di Python.
    public bool TryGetLatest(out GazeMessage msg, out double ageMs, out double latencyMs)
    {
        string json; double recv;
        lock (_lock) { json = _latestJson; recv = _latestRecvMs; }
        msg = null; ageMs = double.MaxValue; latencyMs = 0;
        if (string.IsNullOrEmpty(json)) return false;

        if (!ReferenceEquals(json, _parsedJson))
        {
            _parsed = JsonUtility.FromJson<GazeMessage>(json);
            _parsedJson = json;
        }
        msg = _parsed;
        if (msg == null) return false;
        double now = NowMs();
        ageMs = now - recv;
        latencyMs = now - msg.t_capture_ms;
        return true;
    }
}
