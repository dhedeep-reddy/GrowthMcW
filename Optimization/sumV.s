	.file	"sumV.cpp"
	.text
	.p2align 4
	.globl	_Z3sumRSt6vectorIiSaIiEE
	.type	_Z3sumRSt6vectorIiSaIiEE, @function
_Z3sumRSt6vectorIiSaIiEE:
.LFB2167:
	.cfi_startproc
	endbr64
	movq	(%rdi), %r8
	movq	8(%rdi), %rcx
	subq	%r8, %rcx
	sarq	$2, %rcx
	je	.L9
	leaq	-1(%rcx), %rax
	cmpq	$14, %rax
	jbe	.L10
	movq	%rcx, %rdx
	movq	%r8, %rax
	vpxor	%xmm0, %xmm0, %xmm0
	shrq	$4, %rdx
	salq	$6, %rdx
	addq	%r8, %rdx
	.p2align 4
	.p2align 3
.L4:
	vpmovsxdq	(%rax), %zmm1
	vmovdqu32	(%rax), %zmm3
	addq	$64, %rax
	vpaddq	%zmm0, %zmm1, %zmm1
	vextracti32x8	$0x1, %zmm3, %ymm0
	vpmovsxdq	%ymm0, %zmm0
	vpaddq	%zmm1, %zmm0, %zmm0
	cmpq	%rdx, %rax
	jne	.L4
	vmovdqa	%ymm0, %ymm1
	vextracti64x4	$0x1, %zmm0, %ymm0
	movq	%rcx, %rsi
	vpaddq	%ymm0, %ymm1, %ymm1
	andq	$-16, %rsi
	vmovdqa	%xmm1, %xmm0
	vextracti64x2	$0x1, %ymm1, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vpsrldq	$8, %xmm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovq	%xmm0, %rax
	testb	$15, %cl
	je	.L18
.L3:
	movq	%rcx, %rdi
	subq	%rsi, %rdi
	leaq	-1(%rdi), %rdx
	cmpq	$6, %rdx
	jbe	.L7
	vmovdqu	(%r8,%rsi,4), %ymm2
	vpmovsxdq	(%r8,%rsi,4), %ymm1
	vextracti128	$0x1, %ymm2, %xmm0
	vpmovsxdq	%xmm0, %ymm0
	vpaddq	%ymm0, %ymm1, %ymm1
	vmovdqa	%xmm1, %xmm0
	vextracti64x2	$0x1, %ymm1, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vpsrldq	$8, %xmm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovq	%xmm0, %rdx
	addq	%rdx, %rax
	movq	%rdi, %rdx
	andq	$-8, %rdx
	addq	%rdx, %rsi
	cmpq	%rdx, %rdi
	je	.L18
.L7:
	movslq	(%r8,%rsi,4), %rdi
	leaq	0(,%rsi,4), %rdx
	addq	%rdi, %rax
	leaq	1(%rsi), %rdi
	cmpq	%rcx, %rdi
	jnb	.L18
	movslq	4(%r8,%rdx), %rdi
	addq	%rdi, %rax
	leaq	2(%rsi), %rdi
	cmpq	%rcx, %rdi
	jnb	.L18
	movslq	8(%r8,%rdx), %rdi
	addq	%rdi, %rax
	leaq	3(%rsi), %rdi
	cmpq	%rdi, %rcx
	jbe	.L18
	movslq	12(%r8,%rdx), %rdi
	addq	%rdi, %rax
	leaq	4(%rsi), %rdi
	cmpq	%rdi, %rcx
	jbe	.L18
	movslq	16(%r8,%rdx), %rdi
	addq	%rdi, %rax
	leaq	5(%rsi), %rdi
	cmpq	%rdi, %rcx
	jbe	.L18
	movslq	20(%r8,%rdx), %rdi
	addq	$6, %rsi
	addq	%rdi, %rax
	cmpq	%rsi, %rcx
	jbe	.L18
	movslq	24(%r8,%rdx), %rdx
	addq	%rdx, %rax
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
	xorl	%esi, %esi
	xorl	%eax, %eax
	jmp	.L3
	.cfi_endproc
.LFE2167:
	.size	_Z3sumRSt6vectorIiSaIiEE, .-_Z3sumRSt6vectorIiSaIiEE
	.section	.text.unlikely,"ax",@progbits
.LCOLDB2:
	.section	.text.startup,"ax",@progbits
.LHOTB2:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB2168:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2168
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$400000000, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	pushq	%r12
	andq	$-64, %rsp
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	vpbroadcastd	.LC1(%rip), %zmm0
	movq	%rax, %r12
	movq	%rax, %rdx
	leaq	400000000(%rax), %rcx
	.p2align 4
	.p2align 3
.L21:
	vmovdqu32	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rax, %rcx
	jne	.L21
	vpxor	%xmm0, %xmm0, %xmm0
	.p2align 4
	.p2align 3
.L22:
	vpmovsxdq	(%rdx), %zmm1
	vmovdqu32	(%rdx), %zmm2
	addq	$64, %rdx
	vpaddq	%zmm0, %zmm1, %zmm1
	vextracti32x8	$0x1, %zmm2, %ymm0
	vpmovsxdq	%ymm0, %zmm0
	vpaddq	%zmm1, %zmm0, %zmm0
	cmpq	%rdx, %rcx
	jne	.L22
	vextracti64x4	$0x1, %zmm0, %ymm1
	leaq	_ZSt4cout(%rip), %rdi
	vpaddq	%ymm0, %ymm1, %ymm1
	vmovdqa	%xmm1, %xmm0
	vextracti64x2	$0x1, %ymm1, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vpsrldq	$8, %xmm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovq	%xmm0, %rsi
	vzeroupper
.LEHB1:
	call	_ZNSo9_M_insertIxEERSoT_@PLT
	movq	%rax, %r13
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r13,%rax), %r14
	testq	%r14, %r14
	je	.L33
	cmpb	$0, 56(%r14)
	je	.L24
	movsbl	67(%r14), %esi
.L25:
	movq	%r13, %rdi
	call	_ZNSo3putEc@PLT
	movq	%rax, %rdi
	call	_ZNSo5flushEv@PLT
	movq	%r12, %rdi
	movl	$400000000, %esi
	call	_ZdlPvm@PLT
	leaq	-24(%rbp), %rsp
	xorl	%eax, %eax
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L24:
	.cfi_restore_state
	movq	%r14, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	(%r14), %rax
	movl	$10, %esi
	movq	%r14, %rdi
	call	*48(%rax)
	movsbl	%al, %esi
	jmp	.L25
.L33:
	call	_ZSt16__throw_bad_castv@PLT
.LEHE1:
.L27:
	endbr64
	movq	%rax, %r13
	jmp	.L26
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA2168:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2168-.LLSDACSB2168
.LLSDACSB2168:
	.uleb128 .LEHB0-.LFB2168
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB2168
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L27-.LFB2168
	.uleb128 0
.LLSDACSE2168:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC2168
	.type	main.cold, @function
main.cold:
.LFSB2168:
.L26:
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	movl	$400000000, %esi
	movq	%r12, %rdi
	vzeroupper
	call	_ZdlPvm@PLT
	movq	%r13, %rdi
.LEHB2:
	call	_Unwind_Resume@PLT
.LEHE2:
	.cfi_endproc
.LFE2168:
	.section	.gcc_except_table
.LLSDAC2168:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC2168-.LLSDACSBC2168
.LLSDACSBC2168:
	.uleb128 .LEHB2-.LCOLDB2
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
.LLSDACSEC2168:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE2:
	.section	.text.startup
.LHOTE2:
	.p2align 4
	.type	_GLOBAL__sub_I__Z3sumRSt6vectorIiSaIiEE, @function
_GLOBAL__sub_I__Z3sumRSt6vectorIiSaIiEE:
.LFB2724:
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
.LFE2724:
	.size	_GLOBAL__sub_I__Z3sumRSt6vectorIiSaIiEE, .-_GLOBAL__sub_I__Z3sumRSt6vectorIiSaIiEE
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z3sumRSt6vectorIiSaIiEE
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC1:
	.long	1
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
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
