using UnityEngine;

/// CONTOH pengendali pemain 2D yang membaca IGameInput. Sesuaikan dengan game Anda.
/// Seret EyeGameInput ATAU KeyboardGameInput ke slot "inputSource".
[RequireComponent(typeof(Rigidbody2D))]
public class PlayerController2D : MonoBehaviour
{
    [SerializeField] MonoBehaviour inputSource;   // harus mengimplementasi IGameInput
    [SerializeField] float speed = 6f;
    [SerializeField] float jumpVelocity = 9f;
    [SerializeField] Transform groundCheck;
    [SerializeField] float groundRadius = 0.12f;
    [SerializeField] LayerMask groundMask;

    IGameInput _input;
    Rigidbody2D _rb;

    void Awake()
    {
        _input = inputSource as IGameInput;
        if (_input == null) Debug.LogError("inputSource harus mengimplementasi IGameInput", this);
        _rb = GetComponent<Rigidbody2D>();
    }

    bool Grounded()
    {
        return groundCheck != null && Physics2D.OverlapCircle(groundCheck.position, groundRadius, groundMask);
    }

    void Update()
    {
        if (_input != null && _input.GetActionDown() && Grounded())
            SetVelocity(new Vector2(GetVelocity().x, jumpVelocity));
    }

    void FixedUpdate()
    {
        if (_input == null) return;
        SetVelocity(new Vector2(_input.GetHorizontal() * speed, GetVelocity().y));
    }

    Vector2 GetVelocity()
    {
#if UNITY_6000_0_OR_NEWER
        return _rb.linearVelocity;
#else
        return _rb.velocity;
#endif
    }

    void SetVelocity(Vector2 v)
    {
#if UNITY_6000_0_OR_NEWER
        _rb.linearVelocity = v;
#else
        _rb.velocity = v;
#endif
    }
}
