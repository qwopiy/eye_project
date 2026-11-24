// Antarmuka input yang dibaca game. Game TIDAK perlu tahu apakah inputnya keyboard atau mata.
public interface IGameInput
{
    float GetHorizontal();   // -1 (kiri), 0, +1 (kanan)
    bool GetActionDown();    // true sekali saat aksi (misal lompat) dipicu
    bool GetDown();          // true selama pemain "menunduk" (misal jongkok)
}
