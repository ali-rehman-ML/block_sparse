.text
.global matmul_4x4

# void matmul_4x4(
#     float* A,                x0
#     float* B,                x1
#     int* ind,                x2
#     float* C,                x3
#     size_t N (Bytes),        x4
#     size_t K,                x5
#     size_t loop_count,       x6
#     size_t k,                x7
#     size_t a_idx,            x8
#     size_t b_idx,            x9

matmul_4x4:
    // Save callee-saved registers
    STP     x19, x20, [sp, #-16]!
    STP     x21, x22, [sp, #-16]!
    STP     x29, x30, [sp, #-16]!   // Save frame pointer and return address
    MOV     x29, sp
    STP     q8, q9, [sp, #-32]!     // Save NEON callee-saved registers
    STP     q10, q11, [sp, #-32]!

    // Original logic
    MOV     x22, #128
    MOV     x20, x0
    MOV     x17, #4
    MUL     x18, x6, x17            // x18 = loop_count * 4

    LDR     w19, [x2], #4
    SXTW    x19, w19
    MUL     x19, x19, x22
    ADD     x0, x0, x19

    // Prefetch A
    PRFM    PLDL1KEEP, [x0]

    // Prefetch B
    PRFM    PLDL1KEEP, [x1]

    // Load A block (4x4)
    LDR     q0, [x0], #16
    LDR     q1, [x0], #16
    LDR     q2, [x0], #16
    LDR     q3, [x0], #16

    // Initialize C block (4x4)
    MOVI    v20.4s, #0
    MOVI    v21.4s, #0
    MOVI    v22.4s, #0
    MOVI    v23.4s, #0
    MOVI    v24.4s, #0
    MOVI    v25.4s, #0
    MOVI    v26.4s, #0
    MOVI    v27.4s, #0

    // Load B block partial (2x4)
    PRFM    PLDL1KEEP, [x1]
    LDR     q4, [x1], #16
    LDR     q5, [x1], #16



loop_start:
    PRFM    PLDL1KEEP, [x1]
    LDR     q6, [x1], #16
    LDR     q8, [x0], #16
    FMLA    v20.4s, v4.4s, v0.s[0]
    FMLA    v21.4s, v4.4s, v1.s[0]
    FMLA    v22.4s, v4.4s, v2.s[0]
    FMLA    v23.4s, v4.4s, v3.s[0]

    LDR     q7, [x1], #16
    LDR     q9, [x0], #16
    FMLA    v20.4s, v5.4s, v0.s[1]
    FMLA    v21.4s, v5.4s, v1.s[1]
    FMLA    v22.4s, v5.4s, v2.s[1]
    FMLA    v23.4s, v5.4s, v3.s[1]
    
    LDR     q10, [x0], #16
    FMLA    v20.4s, v6.4s, v0.s[2]
    FMLA    v21.4s, v6.4s, v1.s[2]
    FMLA    v22.4s, v6.4s, v2.s[2]
    FMLA    v23.4s, v6.4s, v3.s[2]

    LDR     q11, [x0], #16
    MOV     x0, x20
    LDR     w19, [x2], #4
    SXTW    x19, w19
    MUL     x19, x19, x22
    ADD     x0, x0, x19

    FMLA    v20.4s, v7.4s, v0.s[3]
    FMLA    v21.4s, v7.4s, v1.s[3]
    FMLA    v22.4s, v7.4s, v2.s[3]
    FMLA    v23.4s, v7.4s, v3.s[3]
    LDR     q0, [x0], #16
    FMLA    v24.4s, v4.4s, v8.s[0]
    FMLA    v25.4s, v4.4s, v9.s[0]
    FMLA    v26.4s, v4.4s, v10.s[0]
    FMLA    v27.4s, v4.4s, v11.s[0]
    LDR     q4, [x1], #16
    LDR     q1, [x0], #16
    FMLA    v24.4s, v5.4s, v8.s[1]
    FMLA    v25.4s, v5.4s, v9.s[1]
    FMLA    v26.4s, v5.4s, v10.s[1]
    FMLA    v27.4s, v5.4s, v11.s[1]
    LDR     q5, [x1], #16
    LDR     q2, [x0], #16
    FMLA    v24.4s, v6.4s, v8.s[2]
    FMLA    v25.4s, v6.4s, v9.s[2]
    FMLA    v26.4s, v6.4s, v10.s[2]
    FMLA    v27.4s, v6.4s, v11.s[2]
    LDR     q3, [x0], #16
    FMLA    v24.4s, v7.4s, v8.s[3]
    FMLA    v25.4s, v7.4s, v9.s[3]
    FMLA    v26.4s, v7.4s, v10.s[3]
    FMLA    v27.4s, v7.4s, v11.s[3]

    // Loop control
    ADD     x7, x7, x17             // k += 4
    CMP     x7, x18                 // Compare k with loop_count * 4
    BLT     loop_start              // Branch if k < loop_count * 4


loop_end:
    PRFM    PLDL1KEEP, [x1]
    LDR     q6, [x1], #16
    LDR     q8, [x0], #16
    FMLA    v20.4s, v4.4s, v0.s[0]
    FMLA    v21.4s, v4.4s, v1.s[0]
    FMLA    v22.4s, v4.4s, v2.s[0]
    FMLA    v23.4s, v4.4s, v3.s[0]

    LDR     q7, [x1], #16
    LDR     q9, [x0], #16
    FMLA    v20.4s, v5.4s, v0.s[1]
    FMLA    v21.4s, v5.4s, v1.s[1]
    FMLA    v22.4s, v5.4s, v2.s[1]
    FMLA    v23.4s, v5.4s, v3.s[1]
    
    LDR     q10, [x0], #16
    FMLA    v20.4s, v6.4s, v0.s[2]
    FMLA    v21.4s, v6.4s, v1.s[2]
    FMLA    v22.4s, v6.4s, v2.s[2]
    FMLA    v23.4s, v6.4s, v3.s[2]
    LDR     q11, [x0], #16


    FMLA    v20.4s, v7.4s, v0.s[3]
    FMLA    v21.4s, v7.4s, v1.s[3]
    FMLA    v22.4s, v7.4s, v2.s[3]
    FMLA    v23.4s, v7.4s, v3.s[3]

    FMLA    v24.4s, v4.4s, v8.s[0]
    FMLA    v25.4s, v4.4s, v9.s[0]
    FMLA    v26.4s, v4.4s, v10.s[0]
    FMLA    v27.4s, v4.4s, v11.s[0]

    FMLA    v24.4s, v5.4s, v8.s[1]
    FMLA    v25.4s, v5.4s, v9.s[1]
    FMLA    v26.4s, v5.4s, v10.s[1]
    FMLA    v27.4s, v5.4s, v11.s[1]

    FMLA    v24.4s, v6.4s, v8.s[2]
    FMLA    v25.4s, v6.4s, v9.s[2]
    FMLA    v26.4s, v6.4s, v10.s[2]
    FMLA    v27.4s, v6.4s, v11.s[2]

    FMLA    v24.4s, v7.4s, v8.s[3]
    FMLA    v25.4s, v7.4s, v9.s[3]
    FMLA    v26.4s, v7.4s, v10.s[3]
    FMLA    v27.4s, v7.4s, v11.s[3]

write_output:
    // Store C in memory (4x4 block)
    STR         q20,  [x3]
    ADD         x3, x3, x5
    STR         q21, [x3]
    ADD         x3, x3, x5
    STR         q22, [x3]
    ADD         x3, x3, x5
    STR         q23, [x3]
    ADD         x3, x3, x5
    STR         q24,  [x3]
    ADD         x3, x3, x5
    STR         q25, [x3]
    ADD         x3, x3, x5
    STR         q26, [x3]
    ADD         x3, x3, x5
    STR         q27, [x3]

    // Restore callee-saved registers
    LDP     q10, q11, [sp], #32
    LDP     q8, q9, [sp], #32
    LDP     x29, x30, [sp], #16
    LDP     x21, x22, [sp], #16
    LDP     x19, x20, [sp], #16
    RET

.section .rodata
msg:
    .asciz "verified\n"