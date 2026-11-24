using UnityEngine;

/// Input keyboard sebagai pembanding/cadangan. Memakai Input Manager lama (Edit > Project Settings > Player >
/// Active Input Handling = Input Manager atau Both). Jika proyek memakai Input System baru, ganti isinya dengan
/// Keyboard.current.
public class KeyboardGameInput : MonoBehaviour, IGameInput
{
    public float GetHorizontal() { return Input.GetAxisRaw("Horizontal"); }
    public bool GetActionDown() { return Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.UpArrow); }
    public bool GetDown() { return Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S); }
}
