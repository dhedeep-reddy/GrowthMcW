	.file	"matVecTil.cpp"
	.text
	.p2align 4
	.globl	_Z4initv
	.type	_Z4initv, @function
_Z4initv:
.LFB7584:
	.cfi_startproc
	endbr64
	vbroadcastsd	.LC1(%rip), %zmm0
	leaq	A(%rip), %rcx
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	8192(%rcx), %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	8388608(%rdx), %rsi
.L3:
	movq	%rcx, %rax
	.p2align 4
	.p2align 3
.L2:
	vmovapd	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rdx, %rax
	jne	.L2
	leaq	8192(%rax), %rdx
	addq	$8192, %rcx
	cmpq	%rsi, %rdx
	jne	.L3
	leaq	B(%rip), %rcx
	leaq	8192(%rcx), %rdx
	leaq	8388608(%rdx), %rsi
.L7:
	movq	%rcx, %rax
	.p2align 4
	.p2align 3
.L6:
	vmovapd	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rdx, %rax
	jne	.L6
	leaq	8192(%rax), %rdx
	addq	$8192, %rcx
	cmpq	%rsi, %rdx
	jne	.L7
	movl	$8388608, %edx
	xorl	%esi, %esi
	leaq	C(%rip), %rdi
	vzeroupper
	popq	%rbp
	.cfi_def_cfa 7, 8
	jmp	memset@PLT
	.cfi_endproc
.LFE7584:
	.size	_Z4initv, .-_Z4initv
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC3:
	.string	"AVX-512 + Tiling Time: "
.LC4:
	.string	" sec\n"
.LC5:
	.string	"C[0][0] = "
.LC6:
	.string	"\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB7585:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-64, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	leaq	A(%rip), %r12
	subq	$64, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	vbroadcastsd	.LC1(%rip), %zmm0
	leaq	8388608(%r12), %rcx
	movq	%r12, %rax
.L22:
	leaq	8192(%rax), %rdx
.L21:
	vmovapd	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rax, %rdx
	jne	.L21
	cmpq	%rcx, %rdx
	je	.L48
	movq	%rdx, %rax
	jmp	.L22
.L48:
	leaq	B(%rip), %r13
	leaq	8388608(%r13), %rcx
	movq	%r13, %rax
.L26:
	leaq	8192(%rax), %rdx
.L25:
	vmovapd	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rax, %rdx
	jne	.L25
	cmpq	%rcx, %rdx
	je	.L49
	movq	%rdx, %rax
	jmp	.L26
.L49:
	leaq	C(%rip), %r14
	movl	$8388608, %edx
	xorl	%esi, %esi
	movq	%r14, %rdi
	vzeroupper
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%r14, %rdi
	movq	%rax, %rbx
	leaq	524288(%r13), %rax
	xorl	%r9d, %r9d
	vmovq	%rax, %xmm9
.L29:
	vmovq	%xmm9, %r13
	movq	%r12, %r11
	xorl	%r10d, %r10d
.L37:
	movq	%r13, %rcx
	movq	%rdi, %r8
	xorl	%r14d, %r14d
.L35:
	leaq	524288(%r8), %rax
	movq	%r11, %r15
	movq	%r8, %rsi
	vmovq	%rax, %xmm11
	leaq	-524288(%rcx), %rax
	vmovq	%rax, %xmm10
.L33:
	vmovapd	(%rsi), %zmm8
	vmovapd	64(%rsi), %zmm7
	vmovq	%xmm10, %rax
	movq	%r15, %rdx
	vmovapd	128(%rsi), %zmm6
	vmovapd	192(%rsi), %zmm5
	vmovapd	256(%rsi), %zmm4
	vmovapd	320(%rsi), %zmm3
	vmovapd	384(%rsi), %zmm2
	vmovapd	448(%rsi), %zmm1
	.p2align 4
	.p2align 3
.L30:
	vbroadcastsd	(%rdx), %zmm0
	addq	$8192, %rax
	addq	$8, %rdx
	vfmadd231pd	-8192(%rax), %zmm0, %zmm8
	vfmadd231pd	-8128(%rax), %zmm0, %zmm7
	vfmadd231pd	-8064(%rax), %zmm0, %zmm6
	vfmadd231pd	-8000(%rax), %zmm0, %zmm5
	vfmadd231pd	-7936(%rax), %zmm0, %zmm4
	vfmadd231pd	-7872(%rax), %zmm0, %zmm3
	vfmadd231pd	-7808(%rax), %zmm0, %zmm2
	vfmadd231pd	-7744(%rax), %zmm0, %zmm1
	cmpq	%rax, %rcx
	jne	.L30
	vmovapd	%zmm8, (%rsi)
	vmovapd	%zmm7, 64(%rsi)
	vmovapd	%zmm6, 128(%rsi)
	vmovq	%xmm11, %rax
	vmovapd	%zmm5, 192(%rsi)
	vmovapd	%zmm4, 256(%rsi)
	vmovapd	%zmm3, 320(%rsi)
	addq	$8192, %rsi
	vmovapd	%zmm2, -7808(%rsi)
	vmovapd	%zmm1, -7744(%rsi)
	addq	$8192, %r15
	cmpq	%rsi, %rax
	jne	.L33
	addq	$64, %r14
	addq	$512, %r8
	addq	$512, %rcx
	cmpq	$1024, %r14
	jne	.L35
	addq	$64, %r10
	addq	$512, %r11
	addq	$524288, %r13
	cmpq	$1024, %r10
	jne	.L37
	addl	$64, %r9d
	addq	$524288, %rdi
	addq	$524288, %r12
	cmpl	$1024, %r9d
	jne	.L29
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	vxorps	%xmm0, %xmm0, %xmm0
	leaq	_ZSt4cout(%rip), %r12
	subq	%rbx, %rax
	movl	$23, %edx
	leaq	.LC3(%rip), %rsi
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	vdivsd	.LC2(%rip), %xmm0, %xmm0
	movq	%r12, %rdi
	vmovsd	%xmm0, -56(%rbp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	-56(%rbp), %xmm0
	movq	%r12, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movl	$5, %edx
	leaq	.LC4(%rip), %rsi
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$10, %edx
	leaq	.LC5(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	C(%rip), %xmm0
	movq	%r12, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movl	$1, %edx
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	addq	$64, %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7585:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I_A, @function
_GLOBAL__sub_I_A:
.LFB8114:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	_ZStL8__ioinit(%rip), %rbp
	movq	%rbp, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	%rbp, %rsi
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	__dso_handle(%rip), %rdx
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE8114:
	.size	_GLOBAL__sub_I_A, .-_GLOBAL__sub_I_A
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_A
	.globl	C
	.bss
	.align 64
	.type	C, @object
	.size	C, 8388608
C:
	.zero	8388608
	.globl	B
	.align 64
	.type	B, @object
	.size	B, 8388608
B:
	.zero	8388608
	.globl	A
	.align 64
	.type	A, @object
	.size	A, 8388608
A:
	.zero	8388608
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	0
	.long	1072693248
	.align 8
.LC2:
	.long	0
	.long	1104006501
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
