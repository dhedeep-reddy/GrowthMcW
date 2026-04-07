	.file	"sum.cpp"
	.text
	.p2align 4
	.globl	_Z3sumPii
	.type	_Z3sumPii, @function
_Z3sumPii:
.LFB1812:
	.cfi_startproc
	endbr64
	movl	%esi, %ecx
	testl	%esi, %esi
	jle	.L9
	leal	-1(%rsi), %eax
	cmpl	$14, %eax
	jbe	.L10
	movl	%esi, %edx
	movq	%rdi, %rax
	vpxor	%xmm0, %xmm0, %xmm0
	shrl	$4, %edx
	decl	%edx
	salq	$6, %rdx
	leaq	64(%rdi,%rdx), %rdx
	.p2align 4
	.p2align 3
.L4:
	vpaddd	(%rax), %zmm0, %zmm0
	addq	$64, %rax
	cmpq	%rax, %rdx
	jne	.L4
	vmovdqa	%ymm0, %ymm1
	vextracti32x8	$0x1, %zmm0, %ymm0
	movl	%ecx, %edx
	vpaddd	%ymm0, %ymm1, %ymm1
	andl	$-16, %edx
	vmovdqa	%xmm1, %xmm0
	vextracti128	$0x1, %ymm1, %xmm1
	movl	%edx, %esi
	vpaddd	%xmm1, %xmm0, %xmm0
	vpsrldq	$8, %xmm0, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vpsrldq	$4, %xmm0, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %eax
	cmpl	%edx, %ecx
	je	.L18
.L3:
	movl	%ecx, %r8d
	subl	%edx, %r8d
	leal	-1(%r8), %r9d
	cmpl	$6, %r9d
	jbe	.L7
	vmovdqu	(%rdi,%rdx,4), %ymm1
	vmovdqa	%xmm1, %xmm0
	vextracti128	$0x1, %ymm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vpsrldq	$8, %xmm0, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vpsrldq	$4, %xmm0, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %edx
	addl	%edx, %eax
	movl	%r8d, %edx
	andl	$-8, %edx
	addl	%edx, %esi
	cmpl	%edx, %r8d
	je	.L18
.L7:
	movslq	%esi, %r8
	addl	(%rdi,%r8,4), %eax
	leaq	0(,%r8,4), %rdx
	leal	1(%rsi), %r8d
	cmpl	%r8d, %ecx
	jle	.L18
	leal	2(%rsi), %r8d
	addl	4(%rdi,%rdx), %eax
	cmpl	%r8d, %ecx
	jle	.L18
	leal	3(%rsi), %r8d
	addl	8(%rdi,%rdx), %eax
	cmpl	%r8d, %ecx
	jle	.L18
	leal	4(%rsi), %r8d
	addl	12(%rdi,%rdx), %eax
	cmpl	%r8d, %ecx
	jle	.L18
	leal	5(%rsi), %r8d
	addl	16(%rdi,%rdx), %eax
	cmpl	%r8d, %ecx
	jle	.L18
	addl	$6, %esi
	addl	20(%rdi,%rdx), %eax
	cmpl	%esi, %ecx
	jle	.L18
	addl	24(%rdi,%rdx), %eax
	vzeroupper
	ret
	.p2align 4
	.p2align 3
.L18:
	vzeroupper
	ret
	.p2align 4
	.p2align 3
.L9:
	xorl	%eax, %eax
	ret
.L10:
	xorl	%edx, %edx
	xorl	%esi, %esi
	xorl	%eax, %eax
	jmp	.L3
	.cfi_endproc
.LFE1812:
	.size	_Z3sumPii, .-_Z3sumPii
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"Sum of elements = "
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB1813:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	leaq	_ZSt4cout(%rip), %rbp
	movl	$18, %edx
	movq	%rbp, %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	leaq	.LC0(%rip), %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rbp, %rdi
	movl	$15, %esi
	call	_ZNSolsEi@PLT
	movq	%rax, %rbp
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rbp,%rax), %r12
	testq	%r12, %r12
	je	.L25
	cmpb	$0, 56(%r12)
	je	.L22
	movsbl	67(%r12), %esi
.L23:
	movq	%rbp, %rdi
	call	_ZNSo3putEc@PLT
	movq	%rax, %rdi
	call	_ZNSo5flushEv@PLT
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L22:
	.cfi_restore_state
	movq	%r12, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	(%r12), %rax
	movl	$10, %esi
	movq	%r12, %rdi
	call	*48(%rax)
	movsbl	%al, %esi
	jmp	.L23
.L25:
	call	_ZSt16__throw_bad_castv@PLT
	.cfi_endproc
.LFE1813:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I__Z3sumPii, @function
_GLOBAL__sub_I__Z3sumPii:
.LFB2303:
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
.LFE2303:
	.size	_GLOBAL__sub_I__Z3sumPii, .-_GLOBAL__sub_I__Z3sumPii
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z3sumPii
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0"
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
