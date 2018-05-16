# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 17.0.4.196 Build 20170411";
# mark_description "-I /home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src -qopenmp -std=c++11 -o example_feautr";
# mark_description "ier.exe -fcode-asm -Faasm.s";
	.file "example_feautrier.cpp"
	.text
..TXTST0:
# -- Begin  main
	.text
# mark_begin;
       .align    16,0x90
	.globl main
# --- main()
main:
..B1.1:                         # Preds ..B1.0
                                # Execution count [0.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0xb, main$$LSDA
..___tag_value_main.17:
..L18:
                                                         #example_feautrier.cpp:39.1
  00000 55               pushq %rbp                             #example_feautrier.cpp:39.1
	.cfi_def_cfa_offset 16
  00001 48 89 e5         movq %rsp, %rbp                        #example_feautrier.cpp:39.1
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
  00004 48 83 e4 80      andq $-128, %rsp                       #example_feautrier.cpp:39.1
  00008 41 54            pushq %r12                             #example_feautrier.cpp:39.1
  0000a 41 55            pushq %r13                             #example_feautrier.cpp:39.1
  0000c 41 56            pushq %r14                             #example_feautrier.cpp:39.1
  0000e 41 57            pushq %r15                             #example_feautrier.cpp:39.1
  00010 53               pushq %rbx                             #example_feautrier.cpp:39.1
  00011 48 81 ec d8 02 
        00 00            subq $728, %rsp                        #example_feautrier.cpp:39.1
  00018 33 f6            xorl %esi, %esi                        #example_feautrier.cpp:39.1
  0001a bf 03 00 00 00   movl $3, %edi                          #example_feautrier.cpp:39.1
  0001f e8 fc ff ff ff   call __intel_new_feature_proc_init     #example_feautrier.cpp:39.1
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xd8, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xf8, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xf0, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xe8, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xe0, 0xff, 0xff, 0xff, 0x22
                                # LOE
..B1.256:                       # Preds ..B1.1
                                # Execution count [0.00e+00]
  00024 0f ae 1c 24      stmxcsr (%rsp)                         #example_feautrier.cpp:39.1
  00028 bf 00 00 00 00   movl $.2.249_2_kmpc_loc_struct_pack.525, %edi #example_feautrier.cpp:39.1
  0002d 33 f6            xorl %esi, %esi                        #example_feautrier.cpp:39.1
  0002f 81 0c 24 40 80 
        00 00            orl $32832, (%rsp)                     #example_feautrier.cpp:39.1
  00036 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:39.1
  00038 0f ae 14 24      ldmxcsr (%rsp)                         #example_feautrier.cpp:39.1
..___tag_value_main.27:
  0003c e8 fc ff ff ff   call __kmpc_begin                      #example_feautrier.cpp:39.1
..___tag_value_main.28:
                                # LOE
..B1.2:                         # Preds ..B1.256
                                # Execution count [1.00e+00]
  00041 48 8b 1d fc ff 
        ff ff            movq _ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rbx #example_feautrier.cpp:43.25
  00048 48 8d bc 24 78 
        01 00 00         lea 376(%rsp), %rdi                    #example_feautrier.cpp:43.25
  00050 48 89 5c 24 78   movq %rbx, 120(%rsp)                   #example_feautrier.cpp:43.17
  00055 e8 fc ff ff ff  #       std::ios_base::ios_base(std::ios_base *const)
        call      _ZNSt8ios_baseC2Ev                            #example_feautrier.cpp:43.25
                                # LOE rbx
..B1.3:                         # Preds ..B1.2
                                # Execution count [2.20e-01]
  0005a 33 d2            xorl %edx, %edx                        #example_feautrier.cpp:43.25
  0005c 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:43.25
  0005e 48 c7 84 24 78 
        01 00 00 10 00 
        00 00            movq $_ZTVSt9basic_iosIcSt11char_traitsIcEE+16, 376(%rsp) #example_feautrier.cpp:43.17
  0006a 48 89 94 24 50 
        02 00 00         movq %rdx, 592(%rsp)                   #example_feautrier.cpp:43.17
  00072 88 84 24 58 02 
        00 00            movb %al, 600(%rsp)                    #example_feautrier.cpp:43.17
  00079 88 84 24 59 02 
        00 00            movb %al, 601(%rsp)                    #example_feautrier.cpp:43.17
  00080 48 89 94 24 60 
        02 00 00         movq %rdx, 608(%rsp)                   #example_feautrier.cpp:43.17
  00088 48 89 94 24 68 
        02 00 00         movq %rdx, 616(%rsp)                   #example_feautrier.cpp:43.17
  00090 48 89 94 24 70 
        02 00 00         movq %rdx, 624(%rsp)                   #example_feautrier.cpp:43.17
  00098 48 89 94 24 78 
        02 00 00         movq %rdx, 632(%rsp)                   #example_feautrier.cpp:43.17
                                # LOE rbx
..B1.5:                         # Preds ..B1.3
                                # Execution count [1.00e+00]
  000a0 48 8b 05 04 00 
        00 00            movq 8+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:43.25
  000a7 33 f6            xorl %esi, %esi                        #example_feautrier.cpp:43.25
  000a9 48 8b 15 0c 00 
        00 00            movq 16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rdx #example_feautrier.cpp:43.25
  000b0 48 89 44 24 78   movq %rax, 120(%rsp)                   #example_feautrier.cpp:43.17
  000b5 48 8b 48 e8      movq -24(%rax), %rcx                   #example_feautrier.cpp:43.25
  000b9 48 89 54 0c 78   movq %rdx, 120(%rsp,%rcx)              #example_feautrier.cpp:43.25
  000be 4c 8b 44 24 78   movq 120(%rsp), %r8                    #example_feautrier.cpp:43.25
  000c3 48 c7 84 24 80 
        00 00 00 00 00 
        00 00            movq $0, 128(%rsp)                     #example_feautrier.cpp:43.17
  000cf 4d 8b 48 e8      movq -24(%r8), %r9                     #example_feautrier.cpp:43.25
  000d3 4a 8d 7c 0c 78   lea 120(%rsp,%r9), %rdi                #example_feautrier.cpp:43.25
..___tag_value_main.29:
  000d8 e8 fc ff ff ff  #       std::basic_ios<char, std::char_traits<char>>::init(std::basic_ios<char, std::char_traits<char>> *, std::basic_streambuf<char, std::char_traits<char>> *)
        call      _ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E #example_feautrier.cpp:43.25
..___tag_value_main.30:
                                # LOE rbx
..B1.6:                         # Preds ..B1.5
                                # Execution count [1.00e+00]
  000dd 48 8b 53 e8      movq -24(%rbx), %rdx                   #example_feautrier.cpp:43.25
  000e1 48 8d bc 24 88 
        00 00 00         lea 136(%rsp), %rdi                    #example_feautrier.cpp:43.25
  000e9 48 8b 05 14 00 
        00 00            movq 24+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:43.25
  000f0 48 89 5f f0      movq %rbx, -16(%rdi)                   #example_feautrier.cpp:43.17
  000f4 48 89 44 14 78   movq %rax, 120(%rsp,%rdx)              #example_feautrier.cpp:43.25
..___tag_value_main.31:
  000f9 e8 fc ff ff ff  #       std::basic_filebuf<char, std::char_traits<char>>::basic_filebuf(std::basic_filebuf<char, std::char_traits<char>> *)
        call      _ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev #example_feautrier.cpp:43.25
..___tag_value_main.32:
                                # LOE
..B1.7:                         # Preds ..B1.6
                                # Execution count [1.00e+00]
  000fe 48 8b 44 24 78   movq 120(%rsp), %rax                   #example_feautrier.cpp:43.25
  00103 48 8d b4 24 88 
        00 00 00         lea 136(%rsp), %rsi                    #example_feautrier.cpp:43.25
  0010b 48 8b 50 e8      movq -24(%rax), %rdx                   #example_feautrier.cpp:43.25
  0010f 48 8d 7c 14 78   lea 120(%rsp,%rdx), %rdi               #example_feautrier.cpp:43.25
..___tag_value_main.33:
  00114 e8 fc ff ff ff  #       std::basic_ios<char, std::char_traits<char>>::init(std::basic_ios<char, std::char_traits<char>> *, std::basic_streambuf<char, std::char_traits<char>> *)
        call      _ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E #example_feautrier.cpp:43.25
..___tag_value_main.34:
                                # LOE
..B1.8:                         # Preds ..B1.7
                                # Execution count [1.00e+00]
  00119 be fe ff ff ff   movl $.L_2__STRING.17, %esi            #example_feautrier.cpp:43.25
  0011e 48 8d bc 24 88 
        00 00 00         lea 136(%rsp), %rdi                    #example_feautrier.cpp:43.25
  00126 ba 08 00 00 00   movl $8, %edx                          #example_feautrier.cpp:43.25
..___tag_value_main.35:
  0012b e8 fc ff ff ff  #       std::basic_filebuf<char, std::char_traits<char>>::open(std::basic_filebuf<char, std::char_traits<char>> *, const char *, std::ios_base::openmode)
        call      _ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode #example_feautrier.cpp:43.25
..___tag_value_main.36:
                                # LOE rax
..B1.10:                        # Preds ..B1.8
                                # Execution count [1.00e+00]
  00130 48 85 c0         testq %rax, %rax                       #example_feautrier.cpp:43.25
  00133 0f 84 f2 09 00 
        00               je ..B1.242 # Prob 12%                 #example_feautrier.cpp:43.25
                                # LOE
..B1.11:                        # Preds ..B1.10
                                # Execution count [8.80e-01]
  00139 48 8b 44 24 78   movq 120(%rsp), %rax                   #example_feautrier.cpp:43.25
  0013e 33 f6            xorl %esi, %esi                        #example_feautrier.cpp:43.25
  00140 48 8b 50 e8      movq -24(%rax), %rdx                   #example_feautrier.cpp:43.25
  00144 48 8d 7c 14 78   lea 120(%rsp,%rdx), %rdi               #example_feautrier.cpp:43.25
..___tag_value_main.37:
  00149 e8 fc ff ff ff  #       std::basic_ios<char, std::char_traits<char>>::clear(std::basic_ios<char, std::char_traits<char>> *, std::ios_base::iostate)
        call      _ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate #example_feautrier.cpp:43.25
..___tag_value_main.38:
                                # LOE
..B1.12:                        # Preds ..B1.11 ..B1.242
                                # Execution count [1.00e+00]
  0014e bf 20 03 00 00   movl $800, %edi                        #example_feautrier.cpp:49.17
..___tag_value_main.39:
  00153 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:49.17
..___tag_value_main.40:
                                # LOE rax
..B1.13:                        # Preds ..B1.12
                                # Execution count [1.00e+00]
  00158 49 89 c6         movq %rax, %r14                        #example_feautrier.cpp:49.17
                                # LOE r14
..B1.14:                        # Preds ..B1.13
                                # Execution count [1.00e+00]
  0015b bf 20 03 00 00   movl $800, %edi                        #example_feautrier.cpp:50.17
..___tag_value_main.41:
  00160 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:50.17
..___tag_value_main.42:
                                # LOE rax r14
..B1.15:                        # Preds ..B1.14
                                # Execution count [1.00e+00]
  00165 49 89 c5         movq %rax, %r13                        #example_feautrier.cpp:50.17
                                # LOE r13 r14
..B1.16:                        # Preds ..B1.15
                                # Execution count [1.00e+00]
  00168 bf 20 03 00 00   movl $800, %edi                        #example_feautrier.cpp:51.17
..___tag_value_main.43:
  0016d e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:51.17
..___tag_value_main.44:
                                # LOE rax r13 r14
..B1.17:                        # Preds ..B1.16
                                # Execution count [1.00e+00]
  00172 48 89 c3         movq %rax, %rbx                        #example_feautrier.cpp:51.17
                                # LOE rbx r13 r14
..B1.18:                        # Preds ..B1.17
                                # Execution count [1.00e+00]
  00175 45 33 ff         xorl %r15d, %r15d                      #example_feautrier.cpp:54.15
  00178 45 33 e4         xorl %r12d, %r12d                      #example_feautrier.cpp:54.15
                                # LOE rbx r12 r13 r14 r15
..B1.19:                        # Preds ..B1.27 ..B1.18
                                # Execution count [1.00e+02]
  0017b 48 8d 7c 24 78   lea 120(%rsp), %rdi                    #example_feautrier.cpp:56.11
  00180 48 8d 74 24 68   lea 104(%rsp), %rsi                    #example_feautrier.cpp:56.11
..___tag_value_main.45:
  00185 e8 fc ff ff ff  #       std::basic_istream<char, std::char_traits<char>>::operator>>(std::basic_istream<char, std::char_traits<char>> *, long &)
        call      _ZNSirsERl                                    #example_feautrier.cpp:56.11
..___tag_value_main.46:
                                # LOE rax rbx r12 r13 r14 r15
..B1.21:                        # Preds ..B1.19
                                # Execution count [1.00e+02]
  0018a 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:56.16
  0018d 4b 8d 34 26      lea (%r14,%r12), %rsi                  #example_feautrier.cpp:56.16
..___tag_value_main.47:
  00191 e8 fc ff ff ff  #       std::basic_istream<char, std::char_traits<char>>::operator>>(std::basic_istream<char, std::char_traits<char>> *, double &)
        call      _ZNSirsERd                                    #example_feautrier.cpp:56.16
..___tag_value_main.48:
                                # LOE rax rbx r12 r13 r14 r15
..B1.23:                        # Preds ..B1.21
                                # Execution count [1.00e+02]
  00196 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:56.24
  00199 4b 8d 74 25 00   lea (%r13,%r12), %rsi                  #example_feautrier.cpp:56.24
..___tag_value_main.49:
  0019e e8 fc ff ff ff  #       std::basic_istream<char, std::char_traits<char>>::operator>>(std::basic_istream<char, std::char_traits<char>> *, double &)
        call      _ZNSirsERd                                    #example_feautrier.cpp:56.24
..___tag_value_main.50:
                                # LOE rax rbx r12 r13 r14 r15
..B1.25:                        # Preds ..B1.23
                                # Execution count [1.00e+02]
  001a3 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:56.35
  001a6 4a 8d 34 23      lea (%rbx,%r12), %rsi                  #example_feautrier.cpp:56.35
..___tag_value_main.51:
  001aa e8 fc ff ff ff  #       std::basic_istream<char, std::char_traits<char>>::operator>>(std::basic_istream<char, std::char_traits<char>> *, double &)
        call      _ZNSirsERd                                    #example_feautrier.cpp:56.35
..___tag_value_main.52:
                                # LOE rbx r12 r13 r14 r15
..B1.27:                        # Preds ..B1.25
                                # Execution count [1.00e+02]
  001af 49 ff c7         incq %r15                              #example_feautrier.cpp:54.30
  001b2 49 83 c4 08      addq $8, %r12                          #example_feautrier.cpp:54.30
  001b6 49 83 ff 64      cmpq $100, %r15                        #example_feautrier.cpp:54.24
  001ba 7c bf            jl ..B1.19 # Prob 99%                  #example_feautrier.cpp:54.24
                                # LOE rbx r12 r13 r14 r15
..B1.28:                        # Preds ..B1.27
                                # Execution count [1.00e+00]: Infreq
  001bc bf 20 03 00 00   movl $800, %edi                        #example_feautrier.cpp:60.18
..___tag_value_main.53:
  001c1 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:60.18
..___tag_value_main.54:
                                # LOE rax rbx r13 r14
..B1.29:                        # Preds ..B1.28
                                # Execution count [1.00e+00]: Infreq
  001c6 48 89 44 24 48   movq %rax, 72(%rsp)                    #example_feautrier.cpp:60.18[spill]
                                # LOE rbx r13 r14
..B1.30:                        # Preds ..B1.29
                                # Execution count [1.00e+00]: Infreq
  001cb bf 20 03 00 00   movl $800, %edi                        #example_feautrier.cpp:61.17
..___tag_value_main.55:
  001d0 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:61.17
..___tag_value_main.56:
                                # LOE rbx r13 r14
..B1.32:                        # Preds ..B1.30
                                # Execution count [9.90e-01]: Infreq
  001d5 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:63.19
  001d7 33 ff            xorl %edi, %edi                        #example_feautrier.cpp:63.32
  001d9 48 89 44 24 30   movq %rax, 48(%rsp)                    #example_feautrier.cpp:63.19
  001de 48 89 44 24 38   movq %rax, 56(%rsp)                    #example_feautrier.cpp:63.19
  001e3 48 89 44 24 40   movq %rax, 64(%rsp)                    #example_feautrier.cpp:63.19
  001e8 e8 fc ff ff ff  #       free(void *)
        call      free                                          #example_feautrier.cpp:63.32
                                # LOE rbx r13 r14
..B1.33:                        # Preds ..B1.32
                                # Execution count [2.00e-02]: Infreq
  001ed bf 80 38 01 00   movl $80000, %edi                      #example_feautrier.cpp:63.32
  001f2 e8 fc ff ff ff  #       malloc(size_t)
        call      malloc                                        #example_feautrier.cpp:63.32
                                # LOE rax rbx r13 r14
..B1.34:                        # Preds ..B1.33
                                # Execution count [1.00e-02]: Infreq
  001f7 48 a9 0f 00 00 
        00               testq $15, %rax                        #example_feautrier.cpp:63.32
  001fd 0f 85 40 08 00 
        00               jne ..B1.214 # Prob 0%                 #example_feautrier.cpp:63.32
                                # LOE rax rbx r13 r14
..B1.35:                        # Preds ..B1.34
                                # Execution count [2.00e-02]: Infreq
  00203 48 85 c0         testq %rax, %rax                       #example_feautrier.cpp:63.32
  00206 0f 84 01 08 00 
        00               je ..B1.209 # Prob 12%                 #example_feautrier.cpp:63.32
                                # LOE rax rbx r13 r14
..B1.36:                        # Preds ..B1.35
                                # Execution count [4.77e-02]: Infreq
  0020c 41 bc 64 00 00 
        00               movl $100, %r12d                       #example_feautrier.cpp:63.32
  00212 4c 89 64 24 38   movq %r12, 56(%rsp)                    #example_feautrier.cpp:63.19
  00217 4c 89 64 24 40   movq %r12, 64(%rsp)                    #example_feautrier.cpp:63.19
  0021c 4c 8b 64 24 68   movq 104(%rsp), %r12                   #example_feautrier.cpp:69.14
  00221 4d 89 e7         movq %r12, %r15                        #example_feautrier.cpp:69.16
  00224 49 c1 ef 3f      shrq $63, %r15                         #example_feautrier.cpp:69.16
  00228 4d 03 e7         addq %r15, %r12                        #example_feautrier.cpp:69.16
  0022b 49 d1 fc         sarq $1, %r12                          #example_feautrier.cpp:69.16
  0022e 4d 89 e7         movq %r12, %r15                        #example_feautrier.cpp:70.14
  00231 49 f7 df         negq %r15                              #example_feautrier.cpp:70.14
  00234 48 89 44 24 30   movq %rax, 48(%rsp)                    #example_feautrier.cpp:63.19
  00239 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #example_feautrier.cpp:72.35
  00243 49 83 c7 64      addq $100, %r15                        #example_feautrier.cpp:70.14
  00247 49 3b c4         cmpq %r12, %rax                        #example_feautrier.cpp:72.35
  0024a 73 31            jae ..B1.42 # Prob 50%                 #example_feautrier.cpp:72.35
                                # LOE rbx r12 r13 r14 r15
..B1.37:                        # Preds ..B1.36 ..B1.42
                                # Execution count [5.80e-01]: Infreq
..___tag_value_main.57:
  0024c e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #example_feautrier.cpp:72.35
..___tag_value_main.58:
                                # LOE rbx r12 r13 r14 r15
..B1.38:                        # Preds ..B1.37
                                # Execution count [5.80e-01]: Infreq
  00251 4a 8d 3c e5 00 
        00 00 00         lea (,%r12,8), %rdi                    #example_feautrier.cpp:72.35
  00259 48 89 7c 24 08   movq %rdi, 8(%rsp)                     #example_feautrier.cpp:72.35[spill]
..___tag_value_main.59:
  0025e e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:72.35
..___tag_value_main.60:
                                # LOE rax rbx r12 r13 r14 r15
..B1.39:                        # Preds ..B1.38
                                # Execution count [5.80e-01]: Infreq
  00263 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:72.35[spill]
                                # LOE rbx r12 r13 r14 r15
..B1.40:                        # Preds ..B1.39
                                # Execution count [5.80e-01]: Infreq
  00267 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #example_feautrier.cpp:73.35
  00271 49 3b c4         cmpq %r12, %rax                        #example_feautrier.cpp:73.35
  00274 72 31            jb ..B1.46 # Prob 50%                  #example_feautrier.cpp:73.35
                                # LOE rbx r12 r13 r14 r15
..B1.41:                        # Preds ..B1.40
                                # Execution count [2.90e-01]: Infreq
  00276 4d 85 e4         testq %r12, %r12                       #example_feautrier.cpp:73.35
  00279 7c 2c            jl ..B1.46 # Prob 16%                  #example_feautrier.cpp:73.35
  0027b eb 2f            jmp ..B1.47 # Prob 100%                #example_feautrier.cpp:73.35
                                # LOE rbx r12 r13 r14 r15
..B1.42:                        # Preds ..B1.36
                                # Execution count [5.00e-01]: Infreq
  0027d 4d 85 e4         testq %r12, %r12                       #example_feautrier.cpp:72.35
  00280 7c ca            jl ..B1.37 # Prob 16%                  #example_feautrier.cpp:72.35
                                # LOE rbx r12 r13 r14 r15
..B1.43:                        # Preds ..B1.42
                                # Execution count [4.20e-01]: Infreq
  00282 4a 8d 3c e5 00 
        00 00 00         lea (,%r12,8), %rdi                    #example_feautrier.cpp:72.35
  0028a 48 89 7c 24 08   movq %rdi, 8(%rsp)                     #example_feautrier.cpp:72.35[spill]
..___tag_value_main.61:
  0028f e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:72.35
..___tag_value_main.62:
                                # LOE rax rbx r12 r13 r14 r15
..B1.44:                        # Preds ..B1.43
                                # Execution count [4.20e-01]: Infreq
  00294 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:72.35[spill]
                                # LOE rbx r12 r13 r14 r15
..B1.45:                        # Preds ..B1.44
                                # Execution count [4.20e-01]: Infreq
  00298 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #example_feautrier.cpp:73.35
  002a2 49 3b c4         cmpq %r12, %rax                        #example_feautrier.cpp:73.35
  002a5 73 05            jae ..B1.47 # Prob 50%                 #example_feautrier.cpp:73.35
                                # LOE rbx r12 r13 r14 r15
..B1.46:                        # Preds ..B1.45 ..B1.41 ..B1.40
                                # Execution count [5.80e-01]: Infreq
..___tag_value_main.63:
  002a7 e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #example_feautrier.cpp:73.35
..___tag_value_main.64:
                                # LOE rbx r12 r13 r14 r15
..B1.47:                        # Preds ..B1.45 ..B1.41 ..B1.46
                                # Execution count [1.00e+00]: Infreq
  002ac 48 8b 7c 24 08   movq 8(%rsp), %rdi                     #example_feautrier.cpp:73.35[spill]
..___tag_value_main.65:
  002b1 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:73.35
..___tag_value_main.66:
                                # LOE rax rbx r12 r13 r14 r15
..B1.48:                        # Preds ..B1.47
                                # Execution count [1.00e+00]: Infreq
  002b6 48 89 44 24 58   movq %rax, 88(%rsp)                    #example_feautrier.cpp:73.35[spill]
                                # LOE rbx r12 r13 r14 r15
..B1.49:                        # Preds ..B1.48
                                # Execution count [1.00e+00]: Infreq
  002bb 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #example_feautrier.cpp:75.36
  002c5 49 3b c7         cmpq %r15, %rax                        #example_feautrier.cpp:75.36
  002c8 73 32            jae ..B1.55 # Prob 50%                 #example_feautrier.cpp:75.36
                                # LOE rbx r12 r13 r14 r15
..B1.50:                        # Preds ..B1.49 ..B1.55
                                # Execution count [5.80e-01]: Infreq
..___tag_value_main.67:
  002ca e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #example_feautrier.cpp:75.36
..___tag_value_main.68:
                                # LOE rbx r12 r13 r14 r15
..B1.51:                        # Preds ..B1.50
                                # Execution count [5.80e-01]: Infreq
  002cf 4a 8d 3c fd 00 
        00 00 00         lea (,%r15,8), %rdi                    #example_feautrier.cpp:75.36
  002d7 48 89 7c 24 08   movq %rdi, 8(%rsp)                     #example_feautrier.cpp:75.36[spill]
..___tag_value_main.69:
  002dc e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:75.36
..___tag_value_main.70:
                                # LOE rax rbx r12 r13 r14 r15
..B1.52:                        # Preds ..B1.51
                                # Execution count [5.80e-01]: Infreq
  002e1 48 89 44 24 50   movq %rax, 80(%rsp)                    #example_feautrier.cpp:75.36[spill]
                                # LOE rbx r12 r13 r14 r15
..B1.53:                        # Preds ..B1.52
                                # Execution count [5.80e-01]: Infreq
  002e6 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #example_feautrier.cpp:76.36
  002f0 49 3b c7         cmpq %r15, %rax                        #example_feautrier.cpp:76.36
  002f3 72 32            jb ..B1.59 # Prob 50%                  #example_feautrier.cpp:76.36
                                # LOE rbx r12 r13 r14 r15
..B1.54:                        # Preds ..B1.53
                                # Execution count [2.90e-01]: Infreq
  002f5 4d 85 ff         testq %r15, %r15                       #example_feautrier.cpp:76.36
  002f8 7c 2d            jl ..B1.59 # Prob 16%                  #example_feautrier.cpp:76.36
  002fa eb 30            jmp ..B1.60 # Prob 100%                #example_feautrier.cpp:76.36
                                # LOE rbx r12 r13 r14 r15
..B1.55:                        # Preds ..B1.49
                                # Execution count [5.00e-01]: Infreq
  002fc 4d 85 ff         testq %r15, %r15                       #example_feautrier.cpp:75.36
  002ff 7c c9            jl ..B1.50 # Prob 16%                  #example_feautrier.cpp:75.36
                                # LOE rbx r12 r13 r14 r15
..B1.56:                        # Preds ..B1.55
                                # Execution count [4.20e-01]: Infreq
  00301 4a 8d 3c fd 00 
        00 00 00         lea (,%r15,8), %rdi                    #example_feautrier.cpp:75.36
  00309 48 89 7c 24 08   movq %rdi, 8(%rsp)                     #example_feautrier.cpp:75.36[spill]
..___tag_value_main.71:
  0030e e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:75.36
..___tag_value_main.72:
                                # LOE rax rbx r12 r13 r14 r15
..B1.57:                        # Preds ..B1.56
                                # Execution count [4.20e-01]: Infreq
  00313 48 89 44 24 50   movq %rax, 80(%rsp)                    #example_feautrier.cpp:75.36[spill]
                                # LOE rbx r12 r13 r14 r15
..B1.58:                        # Preds ..B1.57
                                # Execution count [4.20e-01]: Infreq
  00318 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #example_feautrier.cpp:76.36
  00322 49 3b c7         cmpq %r15, %rax                        #example_feautrier.cpp:76.36
  00325 73 05            jae ..B1.60 # Prob 50%                 #example_feautrier.cpp:76.36
                                # LOE rbx r12 r13 r14 r15
..B1.59:                        # Preds ..B1.58 ..B1.54 ..B1.53
                                # Execution count [5.80e-01]: Infreq
..___tag_value_main.73:
  00327 e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #example_feautrier.cpp:76.36
..___tag_value_main.74:
                                # LOE rbx r12 r13 r14 r15
..B1.60:                        # Preds ..B1.58 ..B1.54 ..B1.59
                                # Execution count [1.00e+00]: Infreq
  0032c 48 8b 7c 24 08   movq 8(%rsp), %rdi                     #example_feautrier.cpp:76.36[spill]
..___tag_value_main.75:
  00331 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #example_feautrier.cpp:76.36
..___tag_value_main.76:
                                # LOE rax rbx r12 r13 r14 r15
..B1.61:                        # Preds ..B1.60
                                # Execution count [1.00e+00]: Infreq
  00336 48 89 44 24 60   movq %rax, 96(%rsp)                    #example_feautrier.cpp:76.36[spill]
                                # LOE rax rbx r12 r13 r14 r15
..B1.62:                        # Preds ..B1.61
                                # Execution count [1.00e+00]: Infreq
  0033b 4d 85 ff         testq %r15, %r15                       #example_feautrier.cpp:79.24
  0033e 0f 8e 92 01 00 
        00               jle ..B1.83 # Prob 50%                 #example_feautrier.cpp:79.24
                                # LOE rax rbx r12 r13 r14 r15
..B1.63:                        # Preds ..B1.62
                                # Execution count [9.00e-01]: Infreq
  00344 49 83 ff 02      cmpq $2, %r15                          #example_feautrier.cpp:79.3
  00348 0f 8c b8 06 00 
        00               jl ..B1.206 # Prob 10%                 #example_feautrier.cpp:79.3
                                # LOE rax rbx r12 r13 r14 r15
..B1.64:                        # Preds ..B1.63
                                # Execution count [9.00e-01]: Infreq
  0034e 49 89 c3         movq %rax, %r11                        #example_feautrier.cpp:79.3
  00351 49 83 e3 0f      andq $15, %r11                         #example_feautrier.cpp:79.3
  00355 74 13            je ..B1.67 # Prob 50%                  #example_feautrier.cpp:79.3
                                # LOE rbx r11 r12 r13 r14 r15
..B1.65:                        # Preds ..B1.64
                                # Execution count [9.00e-01]: Infreq
  00357 49 f7 c3 07 00 
        00 00            testq $7, %r11                         #example_feautrier.cpp:79.3
  0035e 0f 85 a2 06 00 
        00               jne ..B1.206 # Prob 10%                #example_feautrier.cpp:79.3
                                # LOE rbx r12 r13 r14 r15
..B1.66:                        # Preds ..B1.65
                                # Execution count [4.50e-01]: Infreq
  00364 41 bb 01 00 00 
        00               movl $1, %r11d                         #example_feautrier.cpp:79.3
                                # LOE rbx r11 r12 r13 r14 r15
..B1.67:                        # Preds ..B1.66 ..B1.64
                                # Execution count [9.00e-01]: Infreq
  0036a 49 8d 43 02      lea 2(%r11), %rax                      #example_feautrier.cpp:79.3
  0036e 4c 3b f8         cmpq %rax, %r15                        #example_feautrier.cpp:79.3
  00371 0f 8c 8f 06 00 
        00               jl ..B1.206 # Prob 10%                 #example_feautrier.cpp:79.3
                                # LOE rbx r11 r12 r13 r14 r15
..B1.68:                        # Preds ..B1.67
                                # Execution count [1.00e+00]: Infreq
  00377 4c 89 f9         movq %r15, %rcx                        #example_feautrier.cpp:79.3
  0037a 4e 8d 0c e5 00 
        00 00 00         lea (,%r12,8), %r9                     #example_feautrier.cpp:81.20
  00382 49 2b cb         subq %r11, %rcx                        #example_feautrier.cpp:79.3
  00385 4d 89 f2         movq %r14, %r10                        #example_feautrier.cpp:81.20
  00388 48 83 e1 01      andq $1, %rcx                          #example_feautrier.cpp:79.3
  0038c 4d 2b d1         subq %r9, %r10                         #example_feautrier.cpp:81.20
  0038f 48 f7 d9         negq %rcx                              #example_feautrier.cpp:79.3
  00392 49 f7 d9         negq %r9                               #example_feautrier.cpp:82.17
  00395 45 33 c0         xorl %r8d, %r8d                        #example_feautrier.cpp:79.3
  00398 49 03 cf         addq %r15, %rcx                        #example_feautrier.cpp:79.3
  0039b 4d 03 cd         addq %r13, %r9                         #example_feautrier.cpp:82.17
  0039e 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:81.7
  003a0 4d 85 db         testq %r11, %r11                       #example_feautrier.cpp:79.3
  003a3 76 2d            jbe ..B1.72 # Prob 9%                  #example_feautrier.cpp:79.3
                                # LOE rax rcx rbx r8 r9 r10 r11 r12 r13 r14 r15
..B1.69:                        # Preds ..B1.68
                                # Execution count [9.00e-01]: Infreq
  003a5 48 8b 54 24 60   movq 96(%rsp), %rdx                    #[spill]
  003aa 48 8b 74 24 50   movq 80(%rsp), %rsi                    #[spill]
                                # LOE rax rdx rcx rbx rsi r8 r9 r10 r11 r12 r13 r14 r15
..B1.70:                        # Preds ..B1.69 ..B1.70
                                # Execution count [5.00e+00]: Infreq
  003af 49 8b bc c2 18 
        03 00 00         movq 792(%r10,%rax,8), %rdi            #example_feautrier.cpp:81.20
  003b7 4a 89 3c c6      movq %rdi, (%rsi,%r8,8)                #example_feautrier.cpp:81.7
  003bb 49 8b bc c1 18 
        03 00 00         movq 792(%r9,%rax,8), %rdi             #example_feautrier.cpp:82.17
  003c3 48 ff c8         decq %rax                              #example_feautrier.cpp:79.3
  003c6 4a 89 3c c2      movq %rdi, (%rdx,%r8,8)                #example_feautrier.cpp:82.4
  003ca 49 ff c0         incq %r8                               #example_feautrier.cpp:79.3
  003cd 4d 3b c3         cmpq %r11, %r8                         #example_feautrier.cpp:79.3
  003d0 72 dd            jb ..B1.70 # Prob 82%                  #example_feautrier.cpp:79.3
                                # LOE rax rdx rcx rbx rsi r8 r9 r10 r11 r12 r13 r14 r15
..B1.72:                        # Preds ..B1.70 ..B1.68
                                # Execution count [0.00e+00]: Infreq
  003d2 48 8b 54 24 50   movq 80(%rsp), %rdx                    #example_feautrier.cpp:81.7[spill]
  003d7 4c 89 d8         movq %r11, %rax                        #example_feautrier.cpp:79.3
  003da 48 f7 d8         negq %rax                              #example_feautrier.cpp:79.3
  003dd 4a 8d 34 da      lea (%rdx,%r11,8), %rsi                #example_feautrier.cpp:81.7
  003e1 48 f7 c6 0f 00 
        00 00            testq $15, %rsi                        #example_feautrier.cpp:79.3
  003e8 74 57            je ..B1.76 # Prob 60%                  #example_feautrier.cpp:79.3
                                # LOE rax rcx rbx r9 r10 r11 r12 r13 r14 r15
..B1.73:                        # Preds ..B1.72
                                # Execution count [9.00e-01]: Infreq
  003ea 48 8b 54 24 60   movq 96(%rsp), %rdx                    #[spill]
  003ef 48 8b 74 24 50   movq 80(%rsp), %rsi                    #[spill]
  003f4 0f 1f 44 00 00 
        0f 1f 80 00 00 
        00 00            .align    16,0x90
                                # LOE rax rdx rcx rbx rsi r9 r10 r11 r12 r13 r14 r15
..B1.74:                        # Preds ..B1.74 ..B1.73
                                # Execution count [5.00e+00]: Infreq
  00400 f2 41 0f 10 84 
        c2 18 03 00 00   movsd 792(%r10,%rax,8), %xmm0          #example_feautrier.cpp:81.20
  0040a f2 41 0f 10 8c 
        c1 18 03 00 00   movsd 792(%r9,%rax,8), %xmm1           #example_feautrier.cpp:82.17
  00414 66 41 0f 16 8c 
        c1 10 03 00 00   movhpd 784(%r9,%rax,8), %xmm1          #example_feautrier.cpp:82.17
  0041e 66 41 0f 16 84 
        c2 10 03 00 00   movhpd 784(%r10,%rax,8), %xmm0         #example_feautrier.cpp:81.20
  00428 48 83 c0 fe      addq $-2, %rax                         #example_feautrier.cpp:79.3
  0042c 42 0f 11 04 de   movups %xmm0, (%rsi,%r11,8)            #example_feautrier.cpp:81.7
  00431 42 0f 11 0c da   movups %xmm1, (%rdx,%r11,8)            #example_feautrier.cpp:82.4
  00436 49 83 c3 02      addq $2, %r11                          #example_feautrier.cpp:79.3
  0043a 4c 3b d9         cmpq %rcx, %r11                        #example_feautrier.cpp:79.3
  0043d 72 c1            jb ..B1.74 # Prob 82%                  #example_feautrier.cpp:79.3
  0043f eb 49            jmp ..B1.79 # Prob 100%                #example_feautrier.cpp:79.3
                                # LOE rax rdx rcx rbx rsi r9 r10 r11 r12 r13 r14 r15
..B1.76:                        # Preds ..B1.72
                                # Execution count [9.00e-01]: Infreq
  00441 48 8b 54 24 60   movq 96(%rsp), %rdx                    #[spill]
  00446 48 8b 74 24 50   movq 80(%rsp), %rsi                    #[spill]
                                # LOE rax rdx rcx rbx rsi r9 r10 r11 r12 r13 r14 r15
..B1.77:                        # Preds ..B1.77 ..B1.76
                                # Execution count [5.00e+00]: Infreq
  0044b f2 41 0f 10 84 
        c2 18 03 00 00   movsd 792(%r10,%rax,8), %xmm0          #example_feautrier.cpp:81.20
  00455 f2 41 0f 10 8c 
        c1 18 03 00 00   movsd 792(%r9,%rax,8), %xmm1           #example_feautrier.cpp:82.17
  0045f 66 41 0f 16 8c 
        c1 10 03 00 00   movhpd 784(%r9,%rax,8), %xmm1          #example_feautrier.cpp:82.17
  00469 66 41 0f 16 84 
        c2 10 03 00 00   movhpd 784(%r10,%rax,8), %xmm0         #example_feautrier.cpp:81.20
  00473 48 83 c0 fe      addq $-2, %rax                         #example_feautrier.cpp:79.3
  00477 42 0f 11 04 de   movups %xmm0, (%rsi,%r11,8)            #example_feautrier.cpp:81.7
  0047c 42 0f 11 0c da   movups %xmm1, (%rdx,%r11,8)            #example_feautrier.cpp:82.4
  00481 49 83 c3 02      addq $2, %r11                          #example_feautrier.cpp:79.3
  00485 4c 3b d9         cmpq %rcx, %r11                        #example_feautrier.cpp:79.3
  00488 72 c1            jb ..B1.77 # Prob 82%                  #example_feautrier.cpp:79.3
                                # LOE rax rdx rcx rbx rsi r9 r10 r11 r12 r13 r14 r15
..B1.79:                        # Preds ..B1.77 ..B1.74 ..B1.206
                                # Execution count [1.00e+00]: Infreq
  0048a 48 89 ce         movq %rcx, %rsi                        #example_feautrier.cpp:79.3
  0048d 48 f7 de         negq %rsi                              #example_feautrier.cpp:79.3
  00490 49 3b cf         cmpq %r15, %rcx                        #example_feautrier.cpp:79.3
  00493 73 41            jae ..B1.83 # Prob 9%                  #example_feautrier.cpp:79.3
                                # LOE rcx rbx rsi r12 r13 r14 r15
..B1.80:                        # Preds ..B1.79
                                # Execution count [9.00e-01]: Infreq
  00495 4c 89 f2         movq %r14, %rdx                        #example_feautrier.cpp:81.20
  00498 4a 8d 04 e5 00 
        00 00 00         lea (,%r12,8), %rax                    #example_feautrier.cpp:81.20
  004a0 48 2b d0         subq %rax, %rdx                        #example_feautrier.cpp:81.20
  004a3 48 f7 d8         negq %rax                              #example_feautrier.cpp:82.17
  004a6 4c 8b 54 24 60   movq 96(%rsp), %r10                    #example_feautrier.cpp:82.17[spill]
  004ab 49 03 c5         addq %r13, %rax                        #example_feautrier.cpp:82.17
  004ae 4c 8b 5c 24 50   movq 80(%rsp), %r11                    #example_feautrier.cpp:82.17[spill]
                                # LOE rax rdx rcx rbx rsi r10 r11 r12 r13 r14 r15
..B1.81:                        # Preds ..B1.81 ..B1.80
                                # Execution count [5.00e+00]: Infreq
  004b3 4c 8b 84 f2 18 
        03 00 00         movq 792(%rdx,%rsi,8), %r8             #example_feautrier.cpp:81.20
  004bb 4c 8b 8c f0 18 
        03 00 00         movq 792(%rax,%rsi,8), %r9             #example_feautrier.cpp:82.17
  004c3 48 ff ce         decq %rsi                              #example_feautrier.cpp:79.3
  004c6 4d 89 04 cb      movq %r8, (%r11,%rcx,8)                #example_feautrier.cpp:81.7
  004ca 4d 89 0c ca      movq %r9, (%r10,%rcx,8)                #example_feautrier.cpp:82.4
  004ce 48 ff c1         incq %rcx                              #example_feautrier.cpp:79.3
  004d1 49 3b cf         cmpq %r15, %rcx                        #example_feautrier.cpp:79.3
  004d4 72 dd            jb ..B1.81 # Prob 82%                  #example_feautrier.cpp:79.3
                                # LOE rax rdx rcx rbx rsi r10 r11 r12 r13 r14 r15
..B1.83:                        # Preds ..B1.81 ..B1.79 ..B1.62
                                # Execution count [1.00e+00]: Infreq
  004d6 4d 85 e4         testq %r12, %r12                       #example_feautrier.cpp:85.24
  004d9 0f 8e 36 01 00 
        00               jle ..B1.104 # Prob 50%                #example_feautrier.cpp:85.24
                                # LOE rbx r12 r13 r14 r15
..B1.84:                        # Preds ..B1.83
                                # Execution count [9.00e-01]: Infreq
  004df 49 83 fc 02      cmpq $2, %r12                          #example_feautrier.cpp:85.3
  004e3 0f 8c 15 05 00 
        00               jl ..B1.203 # Prob 10%                 #example_feautrier.cpp:85.3
                                # LOE rbx r12 r13 r14 r15
..B1.85:                        # Preds ..B1.84
                                # Execution count [9.00e-01]: Infreq
  004e9 48 8b 74 24 58   movq 88(%rsp), %rsi                    #example_feautrier.cpp:85.3[spill]
  004ee 48 83 e6 0f      andq $15, %rsi                         #example_feautrier.cpp:85.3
  004f2 74 12            je ..B1.88 # Prob 50%                  #example_feautrier.cpp:85.3
                                # LOE rbx rsi r12 r13 r14 r15
..B1.86:                        # Preds ..B1.85
                                # Execution count [9.00e-01]: Infreq
  004f4 48 f7 c6 07 00 
        00 00            testq $7, %rsi                         #example_feautrier.cpp:85.3
  004fb 0f 85 fd 04 00 
        00               jne ..B1.203 # Prob 10%                #example_feautrier.cpp:85.3
                                # LOE rbx r12 r13 r14 r15
..B1.87:                        # Preds ..B1.86
                                # Execution count [4.50e-01]: Infreq
  00501 be 01 00 00 00   movl $1, %esi                          #example_feautrier.cpp:85.3
                                # LOE rbx rsi r12 r13 r14 r15
..B1.88:                        # Preds ..B1.87 ..B1.85
                                # Execution count [9.00e-01]: Infreq
  00506 48 8d 46 02      lea 2(%rsi), %rax                      #example_feautrier.cpp:85.3
  0050a 4c 3b e0         cmpq %rax, %r12                        #example_feautrier.cpp:85.3
  0050d 0f 8c eb 04 00 
        00               jl ..B1.203 # Prob 10%                 #example_feautrier.cpp:85.3
                                # LOE rbx rsi r12 r13 r14 r15
..B1.89:                        # Preds ..B1.88
                                # Execution count [1.00e+00]: Infreq
  00513 4d 89 e0         movq %r12, %r8                         #example_feautrier.cpp:85.3
  00516 4a 8d 04 e5 00 
        00 00 00         lea (,%r12,8), %rax                    #example_feautrier.cpp:87.18
  0051e 4c 2b c6         subq %rsi, %r8                         #example_feautrier.cpp:85.3
  00521 4c 89 f2         movq %r14, %rdx                        #example_feautrier.cpp:87.18
  00524 49 83 e0 01      andq $1, %r8                           #example_feautrier.cpp:85.3
  00528 48 2b d0         subq %rax, %rdx                        #example_feautrier.cpp:87.18
  0052b 49 f7 d8         negq %r8                               #example_feautrier.cpp:85.3
  0052e 48 f7 d8         negq %rax                              #example_feautrier.cpp:88.15
  00531 4d 03 c4         addq %r12, %r8                         #example_feautrier.cpp:85.3
  00534 49 03 c5         addq %r13, %rax                        #example_feautrier.cpp:88.15
  00537 45 33 c9         xorl %r9d, %r9d                        #example_feautrier.cpp:85.3
  0053a 48 85 f6         testq %rsi, %rsi                       #example_feautrier.cpp:85.3
  0053d 76 29            jbe ..B1.93 # Prob 9%                  #example_feautrier.cpp:85.3
                                # LOE rax rdx rbx rsi r8 r9 r12 r13 r14 r15
..B1.90:                        # Preds ..B1.89
                                # Execution count [9.00e-01]: Infreq
  0053f 48 8b 4c 24 58   movq 88(%rsp), %rcx                    #[spill]
  00544 48 8b 3c 24      movq (%rsp), %rdi                      #[spill]
                                # LOE rax rdx rcx rbx rsi rdi r8 r9 r12 r13 r14 r15
..B1.91:                        # Preds ..B1.90 ..B1.91
                                # Execution count [5.00e+00]: Infreq
  00548 4e 8b 94 ca 20 
        03 00 00         movq 800(%rdx,%r9,8), %r10             #example_feautrier.cpp:87.18
  00550 4e 8b 9c c8 20 
        03 00 00         movq 800(%rax,%r9,8), %r11             #example_feautrier.cpp:88.15
  00558 4e 89 14 cf      movq %r10, (%rdi,%r9,8)                #example_feautrier.cpp:87.6
  0055c 4e 89 1c c9      movq %r11, (%rcx,%r9,8)                #example_feautrier.cpp:88.3
  00560 49 ff c1         incq %r9                               #example_feautrier.cpp:85.3
  00563 4c 3b ce         cmpq %rsi, %r9                         #example_feautrier.cpp:85.3
  00566 72 e0            jb ..B1.91 # Prob 82%                  #example_feautrier.cpp:85.3
                                # LOE rax rdx rcx rbx rsi rdi r8 r9 r12 r13 r14 r15
..B1.93:                        # Preds ..B1.91 ..B1.89
                                # Execution count [0.00e+00]: Infreq
  00568 4c 8b 0c 24      movq (%rsp), %r9                       #example_feautrier.cpp:87.6[spill]
  0056c 4d 8d 14 f1      lea (%r9,%rsi,8), %r10                 #example_feautrier.cpp:87.6
  00570 49 f7 c2 0f 00 
        00 00            testq $15, %r10                        #example_feautrier.cpp:85.3
  00577 74 2e            je ..B1.97 # Prob 60%                  #example_feautrier.cpp:85.3
                                # LOE rax rdx rbx rsi r8 r12 r13 r14 r15
..B1.94:                        # Preds ..B1.93
                                # Execution count [9.00e-01]: Infreq
  00579 4c 8b 4c 24 58   movq 88(%rsp), %r9                     #[spill]
  0057e 4c 8b 14 24      movq (%rsp), %r10                      #[spill]
                                # LOE rax rdx rbx rsi r8 r9 r10 r12 r13 r14 r15
..B1.95:                        # Preds ..B1.95 ..B1.94
                                # Execution count [5.00e+00]: Infreq
  00582 0f 10 84 f2 20 
        03 00 00         movups 800(%rdx,%rsi,8), %xmm0         #example_feautrier.cpp:87.18
  0058a 0f 10 8c f0 20 
        03 00 00         movups 800(%rax,%rsi,8), %xmm1         #example_feautrier.cpp:88.15
  00592 41 0f 11 04 f2   movups %xmm0, (%r10,%rsi,8)            #example_feautrier.cpp:87.6
  00597 41 0f 11 0c f1   movups %xmm1, (%r9,%rsi,8)             #example_feautrier.cpp:88.3
  0059c 48 83 c6 02      addq $2, %rsi                          #example_feautrier.cpp:85.3
  005a0 49 3b f0         cmpq %r8, %rsi                         #example_feautrier.cpp:85.3
  005a3 72 dd            jb ..B1.95 # Prob 82%                  #example_feautrier.cpp:85.3
  005a5 eb 2c            jmp ..B1.100 # Prob 100%               #example_feautrier.cpp:85.3
                                # LOE rax rdx rbx rsi r8 r9 r10 r12 r13 r14 r15
..B1.97:                        # Preds ..B1.93
                                # Execution count [9.00e-01]: Infreq
  005a7 4c 8b 4c 24 58   movq 88(%rsp), %r9                     #[spill]
  005ac 4c 8b 14 24      movq (%rsp), %r10                      #[spill]
                                # LOE rax rdx rbx rsi r8 r9 r10 r12 r13 r14 r15
..B1.98:                        # Preds ..B1.98 ..B1.97
                                # Execution count [5.00e+00]: Infreq
  005b0 0f 10 84 f2 20 
        03 00 00         movups 800(%rdx,%rsi,8), %xmm0         #example_feautrier.cpp:87.18
  005b8 0f 10 8c f0 20 
        03 00 00         movups 800(%rax,%rsi,8), %xmm1         #example_feautrier.cpp:88.15
  005c0 41 0f 11 04 f2   movups %xmm0, (%r10,%rsi,8)            #example_feautrier.cpp:87.6
  005c5 41 0f 11 0c f1   movups %xmm1, (%r9,%rsi,8)             #example_feautrier.cpp:88.3
  005ca 48 83 c6 02      addq $2, %rsi                          #example_feautrier.cpp:85.3
  005ce 49 3b f0         cmpq %r8, %rsi                         #example_feautrier.cpp:85.3
  005d1 72 dd            jb ..B1.98 # Prob 82%                  #example_feautrier.cpp:85.3
                                # LOE rax rdx rbx rsi r8 r9 r10 r12 r13 r14 r15
..B1.100:                       # Preds ..B1.98 ..B1.95 ..B1.203
                                # Execution count [1.00e+00]: Infreq
  005d3 4d 3b c4         cmpq %r12, %r8                         #example_feautrier.cpp:85.3
  005d6 73 3d            jae ..B1.104 # Prob 9%                 #example_feautrier.cpp:85.3
                                # LOE rbx r8 r12 r13 r14 r15
..B1.101:                       # Preds ..B1.100
                                # Execution count [9.00e-01]: Infreq
  005d8 4c 89 f2         movq %r14, %rdx                        #example_feautrier.cpp:87.18
  005db 4a 8d 04 e5 00 
        00 00 00         lea (,%r12,8), %rax                    #example_feautrier.cpp:87.18
  005e3 48 2b d0         subq %rax, %rdx                        #example_feautrier.cpp:87.18
  005e6 48 f7 d8         negq %rax                              #example_feautrier.cpp:88.15
  005e9 4c 8b 54 24 58   movq 88(%rsp), %r10                    #example_feautrier.cpp:88.15[spill]
  005ee 49 03 c5         addq %r13, %rax                        #example_feautrier.cpp:88.15
  005f1 4c 8b 1c 24      movq (%rsp), %r11                      #example_feautrier.cpp:88.15[spill]
                                # LOE rax rdx rbx r8 r10 r11 r12 r13 r14 r15
..B1.102:                       # Preds ..B1.102 ..B1.101
                                # Execution count [5.00e+00]: Infreq
  005f5 4a 8b b4 c2 20 
        03 00 00         movq 800(%rdx,%r8,8), %rsi             #example_feautrier.cpp:87.18
  005fd 4e 8b 8c c0 20 
        03 00 00         movq 800(%rax,%r8,8), %r9              #example_feautrier.cpp:88.15
  00605 4b 89 34 c3      movq %rsi, (%r11,%r8,8)                #example_feautrier.cpp:87.6
  00609 4f 89 0c c2      movq %r9, (%r10,%r8,8)                 #example_feautrier.cpp:88.3
  0060d 49 ff c0         incq %r8                               #example_feautrier.cpp:85.3
  00610 4d 3b c4         cmpq %r12, %r8                         #example_feautrier.cpp:85.3
  00613 72 e0            jb ..B1.102 # Prob 82%                 #example_feautrier.cpp:85.3
                                # LOE rax rdx rbx r8 r10 r11 r12 r13 r14 r15
..B1.104:                       # Preds ..B1.102 ..B1.100 ..B1.83
                                # Execution count [1.00e+00]: Infreq
  00615 48 c7 44 24 70 
        00 00 00 00      movq $0, 112(%rsp)                     #example_feautrier.cpp:92.8
                                # LOE rbx r12 r13 r14 r15
..B1.105:                       # Preds ..B1.104
                                # Execution count [1.00e+00]: Infreq
..___tag_value_main.77:
  0061e e8 fc ff ff ff  #       omp_get_wtime()
        call      omp_get_wtime                                 #example_feautrier.cpp:94.8
..___tag_value_main.78:
                                # LOE rbx r12 r13 r14 r15 xmm0
..B1.106:                       # Preds ..B1.105
                                # Execution count [1.00e+00]: Infreq
  00623 0f 28 c8         movaps %xmm0, %xmm1                    #example_feautrier.cpp:94.8
                                # LOE rbx r12 r13 r14 r15 xmm1
..B1.107:                       # Preds ..B1.106
                                # Execution count [1.00e+00]: Infreq
  00626 f2 0f 10 44 24 
        70               movsd 112(%rsp), %xmm0                 #example_feautrier.cpp:94.2
  0062c f2 0f 5c c1      subsd %xmm1, %xmm0                     #example_feautrier.cpp:94.8
  00630 f2 0f 11 44 24 
        70               movsd %xmm0, 112(%rsp)                 #example_feautrier.cpp:94.2
..___tag_value_main.79:
  00636 e8 fc ff ff ff  #       omp_get_wtime()
        call      omp_get_wtime                                 #example_feautrier.cpp:95.8
..___tag_value_main.80:
                                # LOE rbx r12 r13 r14 r15 xmm0
..B1.108:                       # Preds ..B1.107
                                # Execution count [1.00e+00]: Infreq
  0063b 0f 28 c8         movaps %xmm0, %xmm1                    #example_feautrier.cpp:95.8
                                # LOE rbx r12 r13 r14 r15 xmm1
..B1.109:                       # Preds ..B1.108
                                # Execution count [1.00e+00]: Infreq
  0063e f2 0f 10 44 24 
        70               movsd 112(%rsp), %xmm0                 #example_feautrier.cpp:95.2
  00644 bf 00 00 00 00   movl $_ZSt4cout, %edi                  #example_feautrier.cpp:96.8
  00649 be fe ff ff ff   movl $.L_2__STRING.0, %esi             #example_feautrier.cpp:96.8
  0064e f2 0f 58 c1      addsd %xmm1, %xmm0                     #example_feautrier.cpp:95.8
  00652 f2 0f 11 44 24 
        70               movsd %xmm0, 112(%rsp)                 #example_feautrier.cpp:95.2
..___tag_value_main.81:
  00658 e8 fc ff ff ff  #       std::operator<<<std::char_traits<char>>(std::basic_ostream<char, std::char_traits<char>> &, const char *)
        call      _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc #example_feautrier.cpp:96.8
..___tag_value_main.82:
                                # LOE rax rbx r12 r13 r14 r15
..B1.111:                       # Preds ..B1.109
                                # Execution count [1.00e+00]: Infreq
  0065d f2 0f 10 44 24 
        70               movsd 112(%rsp), %xmm0                 #example_feautrier.cpp:96.8
  00663 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:96.8
..___tag_value_main.83:
  00666 e8 fc ff ff ff  #       std::basic_ostream<char, std::char_traits<char>>::operator<<(std::basic_ostream<char, std::char_traits<char>> *, double)
        call      _ZNSolsEd                                     #example_feautrier.cpp:96.8
..___tag_value_main.84:
                                # LOE rax rbx r12 r13 r14 r15
..B1.113:                       # Preds ..B1.111
                                # Execution count [1.00e+00]: Infreq
  0066b 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:96.8
  0066e be 00 00 00 00   movl $_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_, %esi #example_feautrier.cpp:96.8
..___tag_value_main.85:
  00673 e8 fc ff ff ff  #       std::basic_ostream<char, std::char_traits<char>>::operator<<(std::basic_ostream<char, std::char_traits<char>> *, std::basic_ostream<char, std::char_traits<char>>::__ostream_type &(*)(std::basic_ostream<char, std::char_traits<char>>::__ostream_type &))
        call      _ZNSolsEPFRSoS_E                              #example_feautrier.cpp:96.8
..___tag_value_main.86:
                                # LOE rbx r12 r13 r14 r15
..B1.115:                       # Preds ..B1.113
                                # Execution count [1.00e+00]: Infreq
..___tag_value_main.87:
  00678 e8 fc ff ff ff  #       omp_get_wtime()
        call      omp_get_wtime                                 #example_feautrier.cpp:100.8
..___tag_value_main.88:
                                # LOE rbx r12 r13 r14 r15 xmm0
..B1.117:                       # Preds ..B1.115
                                # Execution count [1.00e+00]: Infreq
  0067d f2 0f 10 4c 24 
        70               movsd 112(%rsp), %xmm1                 #example_feautrier.cpp:100.2
  00683 48 8b 44 24 30   movq 48(%rsp), %rax                    #example_feautrier.cpp:102.56
  00688 f2 0f 5c c8      subsd %xmm0, %xmm1                     #example_feautrier.cpp:100.8
  0068c 48 8b 74 24 38   movq 56(%rsp), %rsi                    #example_feautrier.cpp:102.56
  00691 48 8b 54 24 40   movq 64(%rsp), %rdx                    #example_feautrier.cpp:102.56
  00696 f2 0f 11 4c 24 
        70               movsd %xmm1, 112(%rsp)                 #example_feautrier.cpp:100.2
  0069c 48 89 44 24 08   movq %rax, 8(%rsp)                     #example_feautrier.cpp:102.53
  006a1 48 89 74 24 10   movq %rsi, 16(%rsp)                    #example_feautrier.cpp:102.53
  006a6 48 89 54 24 18   movq %rdx, 24(%rsp)                    #example_feautrier.cpp:102.53
  006ab 48 85 c0         testq %rax, %rax                       #example_feautrier.cpp:102.56
  006ae 74 19            je ..B1.121 # Prob 50%                 #example_feautrier.cpp:102.56
                                # LOE rdx rbx rsi r12 r13 r14 r15
..B1.118:                       # Preds ..B1.117
                                # Execution count [3.36e-01]: Infreq
  006b0 48 85 f6         testq %rsi, %rsi                       #example_feautrier.cpp:102.56
  006b3 0f 8c c7 02 00 
        00               jl ..B1.188 # Prob 0%                  #example_feautrier.cpp:102.56
                                # LOE rdx rbx rsi r12 r13 r14 r15
..B1.119:                       # Preds ..B1.118
                                # Execution count [3.32e-01]: Infreq
  006b9 48 85 d2         testq %rdx, %rdx                       #example_feautrier.cpp:102.56
  006bc 0f 8c be 02 00 
        00               jl ..B1.188 # Prob 0%                  #example_feautrier.cpp:102.56
                                # LOE rbx rsi r12 r13 r14 r15
..B1.120:                       # Preds ..B1.119
                                # Execution count [3.31e-01]: Infreq
  006c2 48 89 74 24 20   movq %rsi, 32(%rsp)                    #example_feautrier.cpp:102.53
  006c7 eb 0e            jmp ..B1.122 # Prob 100%               #example_feautrier.cpp:102.53
                                # LOE rbx r12 r13 r14 r15
..B1.121:                       # Preds ..B1.117
                                # Execution count [3.43e-01]: Infreq
  006c9 48 89 74 24 20   movq %rsi, 32(%rsp)                    #example_feautrier.cpp:102.53
  006ce 48 85 f6         testq %rsi, %rsi                       #example_feautrier.cpp:102.56
  006d1 0f 8c 0e 03 00 
        00               jl ..B1.202 # Prob 0%                  #example_feautrier.cpp:102.56
                                # LOE rbx r12 r13 r14 r15
..B1.122:                       # Preds ..B1.121 ..B1.120
                                # Execution count [8.42e-01]: Infreq
  006d7 48 83 c4 e0      addq $-32, %rsp                        #example_feautrier.cpp:102.3
	.cfi_escape 0x2e, 0x20
  006db 4c 89 e7         movq %r12, %rdi                        #example_feautrier.cpp:102.3
  006de 48 8b 44 24 68   movq 104(%rsp), %rax                   #example_feautrier.cpp:102.3[spill]
  006e3 4c 8d 54 24 28   lea 40(%rsp), %r10                     #example_feautrier.cpp:102.3
  006e8 4c 89 f9         movq %r15, %rcx                        #example_feautrier.cpp:102.3
  006eb 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:102.3
  006ef 4c 89 54 24 08   movq %r10, 8(%rsp)                     #example_feautrier.cpp:102.3
  006f4 48 c7 44 24 10 
        64 00 00 00      movq $100, 16(%rsp)                    #example_feautrier.cpp:102.3
  006fd 48 8b 74 24 20   movq 32(%rsp), %rsi                    #example_feautrier.cpp:102.3[spill]
  00702 49 8b 52 50      movq 80(%r10), %rdx                    #example_feautrier.cpp:102.3[spill]
  00706 4d 8b 42 48      movq 72(%r10), %r8                     #example_feautrier.cpp:102.3[spill]
  0070a 4d 8b 4a 58      movq 88(%r10), %r9                     #example_feautrier.cpp:102.3[spill]
..___tag_value_main.90:
  0070e e8 fc ff ff ff  #       feautrier(long, double *, double *, long, double *, double *, double *, Eigen::Ref<Eigen::MatrixXd, 0, Eigen::internal::conditional<false, Eigen::InnerStride<1>, Eigen::OuterStride<-1>>::type> *, long)
        call      _Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl #example_feautrier.cpp:102.3
..___tag_value_main.91:
                                # LOE rbx r13 r14
..B1.123:                       # Preds ..B1.122
                                # Execution count [8.42e-01]: Infreq
  00713 48 83 c4 20      addq $32, %rsp                         #example_feautrier.cpp:102.3
	.cfi_escape 0x2e, 0x00
                                # LOE rbx r13 r14
..B1.124:                       # Preds ..B1.123
                                # Execution count [1.00e+00]: Infreq
..___tag_value_main.93:
  00717 e8 fc ff ff ff  #       omp_get_wtime()
        call      omp_get_wtime                                 #example_feautrier.cpp:104.8
..___tag_value_main.94:
                                # LOE rbx r13 r14 xmm0
..B1.125:                       # Preds ..B1.124
                                # Execution count [1.00e+00]: Infreq
  0071c 0f 28 c8         movaps %xmm0, %xmm1                    #example_feautrier.cpp:104.8
                                # LOE rbx r13 r14 xmm1
..B1.126:                       # Preds ..B1.125
                                # Execution count [1.00e+00]: Infreq
  0071f f2 0f 10 44 24 
        70               movsd 112(%rsp), %xmm0                 #example_feautrier.cpp:104.2
  00725 bf 00 00 00 00   movl $_ZSt4cout, %edi                  #example_feautrier.cpp:106.8
  0072a be fe ff ff ff   movl $.L_2__STRING.0, %esi             #example_feautrier.cpp:106.8
  0072f f2 0f 58 c1      addsd %xmm1, %xmm0                     #example_feautrier.cpp:104.8
  00733 f2 0f 11 44 24 
        70               movsd %xmm0, 112(%rsp)                 #example_feautrier.cpp:104.2
..___tag_value_main.95:
  00739 e8 fc ff ff ff  #       std::operator<<<std::char_traits<char>>(std::basic_ostream<char, std::char_traits<char>> &, const char *)
        call      _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc #example_feautrier.cpp:106.8
..___tag_value_main.96:
                                # LOE rax rbx r13 r14
..B1.128:                       # Preds ..B1.126
                                # Execution count [1.00e+00]: Infreq
  0073e f2 0f 10 44 24 
        70               movsd 112(%rsp), %xmm0                 #example_feautrier.cpp:106.8
  00744 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:106.8
..___tag_value_main.97:
  00747 e8 fc ff ff ff  #       std::basic_ostream<char, std::char_traits<char>>::operator<<(std::basic_ostream<char, std::char_traits<char>> *, double)
        call      _ZNSolsEd                                     #example_feautrier.cpp:106.8
..___tag_value_main.98:
                                # LOE rax rbx r13 r14
..B1.130:                       # Preds ..B1.128
                                # Execution count [1.00e+00]: Infreq
  0074c 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:106.8
  0074f be 00 00 00 00   movl $_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_, %esi #example_feautrier.cpp:106.8
..___tag_value_main.99:
  00754 e8 fc ff ff ff  #       std::basic_ostream<char, std::char_traits<char>>::operator<<(std::basic_ostream<char, std::char_traits<char>> *, std::basic_ostream<char, std::char_traits<char>>::__ostream_type &(*)(std::basic_ostream<char, std::char_traits<char>>::__ostream_type &))
        call      _ZNSolsEPFRSoS_E                              #example_feautrier.cpp:106.8
..___tag_value_main.100:
                                # LOE rbx r13 r14
..B1.132:                       # Preds ..B1.130
                                # Execution count [1.00e+00]: Infreq
  00759 48 83 3c 24 00   cmpq $0, (%rsp)                        #example_feautrier.cpp:111.13[spill]
  0075e 74 09            je ..B1.134 # Prob 32%                 #example_feautrier.cpp:111.13
                                # LOE rbx r13 r14
..B1.133:                       # Preds ..B1.132
                                # Execution count [6.74e-01]: Infreq
  00760 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:111.3[spill]
  00764 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:111.3
                                # LOE rbx r13 r14
..B1.134:                       # Preds ..B1.133 ..B1.132
                                # Execution count [1.00e+00]: Infreq
  00769 48 83 7c 24 58 
        00               cmpq $0, 88(%rsp)                      #example_feautrier.cpp:112.12[spill]
  0076f 74 0a            je ..B1.136 # Prob 32%                 #example_feautrier.cpp:112.12
                                # LOE rbx r13 r14
..B1.135:                       # Preds ..B1.134
                                # Execution count [6.74e-01]: Infreq
  00771 48 8b 7c 24 58   movq 88(%rsp), %rdi                    #example_feautrier.cpp:112.2[spill]
  00776 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:112.2
                                # LOE rbx r13 r14
..B1.136:                       # Preds ..B1.135 ..B1.134
                                # Execution count [1.00e+00]: Infreq
  0077b 48 83 7c 24 50 
        00               cmpq $0, 80(%rsp)                      #example_feautrier.cpp:113.12[spill]
  00781 74 0a            je ..B1.138 # Prob 32%                 #example_feautrier.cpp:113.12
                                # LOE rbx r13 r14
..B1.137:                       # Preds ..B1.136
                                # Execution count [6.74e-01]: Infreq
  00783 48 8b 7c 24 50   movq 80(%rsp), %rdi                    #example_feautrier.cpp:113.2[spill]
  00788 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:113.2
                                # LOE rbx r13 r14
..B1.138:                       # Preds ..B1.137 ..B1.136
                                # Execution count [1.00e+00]: Infreq
  0078d 48 83 7c 24 60 
        00               cmpq $0, 96(%rsp)                      #example_feautrier.cpp:114.12[spill]
  00793 74 0a            je ..B1.140 # Prob 32%                 #example_feautrier.cpp:114.12
                                # LOE rbx r13 r14
..B1.139:                       # Preds ..B1.138
                                # Execution count [6.74e-01]: Infreq
  00795 48 8b 7c 24 60   movq 96(%rsp), %rdi                    #example_feautrier.cpp:114.2[spill]
  0079a e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:114.2
                                # LOE rbx r13 r14
..B1.140:                       # Preds ..B1.139 ..B1.138
                                # Execution count [1.00e+00]: Infreq
  0079f 48 83 7c 24 48 
        00               cmpq $0, 72(%rsp)                      #example_feautrier.cpp:116.13[spill]
  007a5 74 0a            je ..B1.142 # Prob 32%                 #example_feautrier.cpp:116.13
                                # LOE rbx r13 r14
..B1.141:                       # Preds ..B1.140
                                # Execution count [6.74e-01]: Infreq
  007a7 48 8b 7c 24 48   movq 72(%rsp), %rdi                    #example_feautrier.cpp:116.3[spill]
  007ac e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:116.3
                                # LOE rbx r13 r14
..B1.142:                       # Preds ..B1.141 ..B1.140
                                # Execution count [1.00e+00]: Infreq
  007b1 4d 85 f6         testq %r14, %r14                       #example_feautrier.cpp:118.13
  007b4 74 08            je ..B1.144 # Prob 32%                 #example_feautrier.cpp:118.13
                                # LOE rbx r13 r14
..B1.143:                       # Preds ..B1.142
                                # Execution count [6.74e-01]: Infreq
  007b6 4c 89 f7         movq %r14, %rdi                        #example_feautrier.cpp:118.3
  007b9 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:118.3
                                # LOE rbx r13
..B1.144:                       # Preds ..B1.143 ..B1.142
                                # Execution count [1.00e+00]: Infreq
  007be 4d 85 ed         testq %r13, %r13                       #example_feautrier.cpp:119.13
  007c1 74 08            je ..B1.146 # Prob 32%                 #example_feautrier.cpp:119.13
                                # LOE rbx r13
..B1.145:                       # Preds ..B1.144
                                # Execution count [6.74e-01]: Infreq
  007c3 4c 89 ef         movq %r13, %rdi                        #example_feautrier.cpp:119.3
  007c6 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:119.3
                                # LOE rbx
..B1.146:                       # Preds ..B1.145 ..B1.144
                                # Execution count [1.00e+00]: Infreq
  007cb 48 85 db         testq %rbx, %rbx                       #example_feautrier.cpp:120.13
  007ce 74 08            je ..B1.148 # Prob 32%                 #example_feautrier.cpp:120.13
                                # LOE rbx
..B1.147:                       # Preds ..B1.146
                                # Execution count [6.74e-01]: Infreq
  007d0 48 89 df         movq %rbx, %rdi                        #example_feautrier.cpp:120.3
  007d3 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #example_feautrier.cpp:120.3
                                # LOE
..B1.148:                       # Preds ..B1.147 ..B1.146
                                # Execution count [1.00e+00]: Infreq
  007d8 48 8d 7c 24 30   lea 48(%rsp), %rdi                     #example_feautrier.cpp:123.2
..___tag_value_main.101:
  007dd e8 fc ff ff ff  #       Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0>::~DenseStorage(Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0> *)
        call      _ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev #example_feautrier.cpp:123.2
..___tag_value_main.102:
                                # LOE
..B1.149:                       # Preds ..B1.148
                                # Execution count [1.00e+00]: Infreq
  007e2 48 8b 05 fc ff 
        ff ff            movq _ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:123.2
  007e9 48 8d bc 24 88 
        00 00 00         lea 136(%rsp), %rdi                    #example_feautrier.cpp:123.2
  007f1 48 8b 15 14 00 
        00 00            movq 24+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rdx #example_feautrier.cpp:123.2
  007f8 48 89 47 f0      movq %rax, -16(%rdi)                   #example_feautrier.cpp:123.2
  007fc 48 8b 48 e8      movq -24(%rax), %rcx                   #example_feautrier.cpp:123.2
  00800 48 89 54 0c 78   movq %rdx, 120(%rsp,%rcx)              #example_feautrier.cpp:123.2
  00805 48 c7 07 10 00 
        00 00            movq $_ZTVSt13basic_filebufIcSt11char_traitsIcEE+16, (%rdi) #example_feautrier.cpp:123.2
..___tag_value_main.103:
  0080c e8 fc ff ff ff  #       std::basic_filebuf<char, std::char_traits<char>>::close(std::basic_filebuf<char, std::char_traits<char>> *)
        call      _ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv #example_feautrier.cpp:123.2
..___tag_value_main.104:
                                # LOE
..B1.151:                       # Preds ..B1.149
                                # Execution count [1.00e+00]: Infreq
  00811 48 8d bc 24 f0 
        00 00 00         lea 240(%rsp), %rdi                    #example_feautrier.cpp:123.2
  00819 e8 fc ff ff ff  #       std::__basic_file<char>::~__basic_file(std::__basic_file<char> *)
        call      _ZNSt12__basic_fileIcED1Ev                    #example_feautrier.cpp:123.2
                                # LOE
..B1.152:                       # Preds ..B1.151
                                # Execution count [1.00e+00]: Infreq
  0081e 48 c7 84 24 88 
        00 00 00 10 00 
        00 00            movq $_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16, 136(%rsp) #example_feautrier.cpp:123.2
  0082a 48 8d bc 24 c0 
        00 00 00         lea 192(%rsp), %rdi                    #example_feautrier.cpp:123.2
  00832 e8 fc ff ff ff  #       std::locale::~locale(std::locale *)
        call      _ZNSt6localeD1Ev                              #example_feautrier.cpp:123.2
                                # LOE
..B1.156:                       # Preds ..B1.152
                                # Execution count [1.00e+00]: Infreq
  00837 48 8b 05 04 00 
        00 00            movq 8+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:123.2
  0083e 48 8b 15 0c 00 
        00 00            movq 16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rdx #example_feautrier.cpp:123.2
  00845 48 89 44 24 78   movq %rax, 120(%rsp)                   #example_feautrier.cpp:123.2
  0084a 48 8b 48 e8      movq -24(%rax), %rcx                   #example_feautrier.cpp:123.2
  0084e 48 89 54 0c 78   movq %rdx, 120(%rsp,%rcx)              #example_feautrier.cpp:123.2
  00853 48 c7 84 24 80 
        00 00 00 00 00 
        00 00            movq $0, 128(%rsp)                     #example_feautrier.cpp:123.2
                                # LOE
..B1.159:                       # Preds ..B1.156
                                # Execution count [2.20e-01]: Infreq
  0085f 48 c7 84 24 78 
        01 00 00 10 00 
        00 00            movq $_ZTVSt9basic_iosIcSt11char_traitsIcEE+16, 376(%rsp) #example_feautrier.cpp:123.2
  0086b 48 8d bc 24 78 
        01 00 00         lea 376(%rsp), %rdi                    #example_feautrier.cpp:123.2
..___tag_value_main.105:
  00873 e8 fc ff ff ff  #       std::ios_base::~ios_base(std::ios_base *const)
        call      _ZNSt8ios_baseD2Ev                            #example_feautrier.cpp:123.2
..___tag_value_main.106:
                                # LOE
..B1.160:                       # Preds ..B1.159
                                # Execution count [2.20e-01]: Infreq
  00878 bf 30 00 00 00   movl $.2.249_2_kmpc_loc_struct_pack.784, %edi #example_feautrier.cpp:123.9
  0087d 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:123.9
..___tag_value_main.107:
  0087f e8 fc ff ff ff   call __kmpc_end                        #example_feautrier.cpp:123.9
..___tag_value_main.108:
                                # LOE
..B1.161:                       # Preds ..B1.160
                                # Execution count [1.00e+00]: Infreq
  00884 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:123.9
  00886 48 81 c4 d8 02 
        00 00            addq $728, %rsp                        #example_feautrier.cpp:123.9
	.cfi_restore 3
  0088d 5b               popq %rbx                              #example_feautrier.cpp:123.9
	.cfi_restore 15
  0088e 41 5f            popq %r15                              #example_feautrier.cpp:123.9
	.cfi_restore 14
  00890 41 5e            popq %r14                              #example_feautrier.cpp:123.9
	.cfi_restore 13
  00892 41 5d            popq %r13                              #example_feautrier.cpp:123.9
	.cfi_restore 12
  00894 41 5c            popq %r12                              #example_feautrier.cpp:123.9
  00896 48 89 ec         movq %rbp, %rsp                        #example_feautrier.cpp:123.9
  00899 5d               popq %rbp                              #example_feautrier.cpp:123.9
	.cfi_def_cfa 7, 8
	.cfi_restore 6
  0089a c3               ret                                    #example_feautrier.cpp:123.9
	.cfi_def_cfa 6, 16
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xd8, 0xff, 0xff, 0xff, 0x22
	.cfi_offset 6, -16
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xf8, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xf0, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xe8, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0x80, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xe0, 0xff, 0xff, 0xff, 0x22
                                # LOE
..___tag_value_main.13:
..B1.162:                       # Preds ..B1.56 ..B1.43 ..B1.130 ..B1.128 ..B1.126
                                #       ..B1.124 ..B1.122 ..B1.115 ..B1.113 ..B1.111
                                #       ..B1.109 ..B1.107 ..B1.105 ..B1.60 ..B1.59
                                #       ..B1.51 ..B1.50 ..B1.47 ..B1.46 ..B1.38
                                #       ..B1.37
                                # Execution count [0.00e+00]: Infreq
  0089b 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:63.32
                                # LOE
..B1.163:                       # Preds ..B1.162
                                # Execution count [0.00e+00]: Infreq
  0089f 48 8d 7c 24 30   lea 48(%rsp), %rdi                     #example_feautrier.cpp:63.32
..___tag_value_main.123:
  008a4 e8 fc ff ff ff  #       Eigen::Matrix<double, -1, -1, 0, -1, -1>::~Matrix(Eigen::Matrix<double, -1, -1, 0, -1, -1> *)
        call      _ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev #example_feautrier.cpp:63.32
..___tag_value_main.124:
                                # LOE
..B1.165:                       # Preds ..B1.212 ..B1.163 ..B1.215
                                # Execution count [0.00e+00]: Infreq
  008a9 48 8b 05 fc ff 
        ff ff            movq _ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:43.25
  008b0 48 8d bc 24 88 
        00 00 00         lea 136(%rsp), %rdi                    #example_feautrier.cpp:43.25
  008b8 48 8b 15 14 00 
        00 00            movq 24+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rdx #example_feautrier.cpp:43.25
  008bf 48 89 47 f0      movq %rax, -16(%rdi)                   #example_feautrier.cpp:43.25
  008c3 48 8b 48 e8      movq -24(%rax), %rcx                   #example_feautrier.cpp:43.25
  008c7 48 89 54 0c 78   movq %rdx, 120(%rsp,%rcx)              #example_feautrier.cpp:43.25
  008cc 48 c7 07 10 00 
        00 00            movq $_ZTVSt13basic_filebufIcSt11char_traitsIcEE+16, (%rdi) #example_feautrier.cpp:43.25
..___tag_value_main.125:
  008d3 e8 fc ff ff ff  #       std::basic_filebuf<char, std::char_traits<char>>::close(std::basic_filebuf<char, std::char_traits<char>> *)
        call      _ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv #example_feautrier.cpp:43.25
..___tag_value_main.126:
                                # LOE
..B1.167:                       # Preds ..B1.165
                                # Execution count [0.00e+00]: Infreq
  008d8 48 8d bc 24 f0 
        00 00 00         lea 240(%rsp), %rdi                    #example_feautrier.cpp:43.25
  008e0 e8 fc ff ff ff  #       std::__basic_file<char>::~__basic_file(std::__basic_file<char> *)
        call      _ZNSt12__basic_fileIcED1Ev                    #example_feautrier.cpp:43.25
                                # LOE
..B1.168:                       # Preds ..B1.167
                                # Execution count [0.00e+00]: Infreq
  008e5 48 8d 84 24 88 
        00 00 00         lea 136(%rsp), %rax                    #example_feautrier.cpp:43.25
  008ed 48 c7 00 10 00 
        00 00            movq $_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.25
  008f4 48 8d 78 38      lea 56(%rax), %rdi                     #example_feautrier.cpp:43.25
  008f8 e8 fc ff ff ff  #       std::locale::~locale(std::locale *)
        call      _ZNSt6localeD1Ev                              #example_feautrier.cpp:43.25
                                # LOE
..B1.172:                       # Preds ..B1.168
                                # Execution count [0.00e+00]: Infreq
  008fd 48 8b 05 04 00 
        00 00            movq 8+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:43.25
  00904 48 8b 15 0c 00 
        00 00            movq 16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rdx #example_feautrier.cpp:43.25
  0090b 48 8d 5c 24 78   lea 120(%rsp), %rbx                    #example_feautrier.cpp:43.25
  00910 48 89 03         movq %rax, (%rbx)                      #example_feautrier.cpp:43.25
  00913 48 8b 48 e8      movq -24(%rax), %rcx                   #example_feautrier.cpp:43.25
  00917 48 89 54 0c 78   movq %rdx, 120(%rsp,%rcx)              #example_feautrier.cpp:43.25
  0091c 48 c7 43 08 00 
        00 00 00         movq $0, 8(%rbx)                       #example_feautrier.cpp:43.25
                                # LOE
..B1.175:                       # Preds ..B1.172
                                # Execution count [0.00e+00]: Infreq
  00924 48 8d 84 24 78 
        01 00 00         lea 376(%rsp), %rax                    #example_feautrier.cpp:43.25
  0092c 48 c7 00 10 00 
        00 00            movq $_ZTVSt9basic_iosIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.25
  00933 48 8d 38         lea (%rax), %rdi                       #example_feautrier.cpp:43.25
..___tag_value_main.127:
  00936 e8 fc ff ff ff  #       std::ios_base::~ios_base(std::ios_base *const)
        call      _ZNSt8ios_baseD2Ev                            #example_feautrier.cpp:43.25
..___tag_value_main.128:
  0093b eb 74            jmp ..B1.193 # Prob 100%               #example_feautrier.cpp:43.25
                                # LOE
..___tag_value_main.11:
..B1.177:                       # Preds ..B1.175
                                # Execution count [0.00e+00]: Infreq
  0093d 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
                                # LOE
..B1.178:                       # Preds ..B1.185 ..B1.177
                                # Execution count [0.00e+00]: Infreq
  00941 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:43.25
  00943 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:43.25
..___tag_value_main.129:
  00947 e8 fc ff ff ff   call __cxa_call_unexpected             #example_feautrier.cpp:43.25
..___tag_value_main.130:
                                # LOE
..B1.179:                       # Preds ..B1.178
                                # Execution count [0.00e+00]: Infreq
  0094c 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:43.25
..___tag_value_main.131:
  00950 e8 fc ff ff ff   call _Unwind_Resume                    #example_feautrier.cpp:43.25
..___tag_value_main.132:
                                # LOE
..___tag_value_main.9:
..___tag_value_main.10:
..B1.183:                       # Preds ..B1.165
                                # Execution count [0.00e+00]: Infreq
  00955 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
                                # LOE
..B1.184:                       # Preds ..B1.183
                                # Execution count [0.00e+00]: Infreq
  00959 48 8d bc 24 f0 
        00 00 00         lea 240(%rsp), %rdi                    #example_feautrier.cpp:43.25
  00961 e8 fc ff ff ff  #       std::__basic_file<char>::~__basic_file(std::__basic_file<char> *)
        call      _ZNSt12__basic_fileIcED1Ev                    #example_feautrier.cpp:43.25
                                # LOE
..B1.185:                       # Preds ..B1.184
                                # Execution count [0.00e+00]: Infreq
  00966 48 8d 84 24 88 
        00 00 00         lea 136(%rsp), %rax                    #example_feautrier.cpp:43.25
  0096e 48 c7 00 10 00 
        00 00            movq $_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.25
  00975 48 8d 78 38      lea 56(%rax), %rdi                     #example_feautrier.cpp:43.25
  00979 e8 fc ff ff ff  #       std::locale::~locale(std::locale *)
        call      _ZNSt6localeD1Ev                              #example_feautrier.cpp:43.25
  0097e eb c1            jmp ..B1.178 # Prob 100%               #example_feautrier.cpp:43.25
                                # LOE
..B1.188:                       # Preds ..B1.118 ..B1.119
                                # Execution count [1.66e-03]: Infreq
  00980 48 c7 44 24 20 
        00 00 00 00      movq $0, 32(%rsp)                      #example_feautrier.cpp:102.53
                                # LOE
..B1.189:                       # Preds ..B1.188
                                # Execution count [4.99e-03]: Infreq
  00989 bf fe ff ff ff   movl $.L_2__STRING.8, %edi             #example_feautrier.cpp:102.56
  0098e be fe ff ff ff   movl $.L_2__STRING.9, %esi             #example_feautrier.cpp:102.56
  00993 ba af 00 00 00   movl $175, %edx                        #example_feautrier.cpp:102.56
  00998 b9 00 00 00 00   movl $__$U7, %ecx                      #example_feautrier.cpp:102.56
  0099d e8 fc ff ff ff  #       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #example_feautrier.cpp:102.56
                                # LOE
..___tag_value_main.16:
..B1.191:                       # Preds ..B1.159
                                # Execution count [0.00e+00]: Infreq
  009a2 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:123.2
                                # LOE
..B1.192:                       # Preds ..B1.199 ..B1.191
                                # Execution count [0.00e+00]: Infreq
  009a6 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:123.2
  009a8 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:123.2
..___tag_value_main.133:
  009ac e8 fc ff ff ff   call __cxa_call_unexpected             #example_feautrier.cpp:123.2
..___tag_value_main.134:
                                # LOE
..B1.193:                       # Preds ..B1.229 ..B1.175 ..B1.192
                                # Execution count [0.00e+00]: Infreq
  009b1 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:123.2
..___tag_value_main.135:
  009b5 e8 fc ff ff ff   call _Unwind_Resume                    #example_feautrier.cpp:123.2
..___tag_value_main.136:
                                # LOE
..___tag_value_main.14:
..___tag_value_main.15:
..B1.197:                       # Preds ..B1.149
                                # Execution count [0.00e+00]: Infreq
  009ba 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:123.2
                                # LOE
..B1.198:                       # Preds ..B1.197
                                # Execution count [0.00e+00]: Infreq
  009be 48 8d bc 24 f0 
        00 00 00         lea 240(%rsp), %rdi                    #example_feautrier.cpp:123.2
  009c6 e8 fc ff ff ff  #       std::__basic_file<char>::~__basic_file(std::__basic_file<char> *)
        call      _ZNSt12__basic_fileIcED1Ev                    #example_feautrier.cpp:123.2
                                # LOE
..B1.199:                       # Preds ..B1.198
                                # Execution count [0.00e+00]: Infreq
  009cb 48 8d 84 24 88 
        00 00 00         lea 136(%rsp), %rax                    #example_feautrier.cpp:123.2
  009d3 48 c7 00 10 00 
        00 00            movq $_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:123.2
  009da 48 8d 78 38      lea 56(%rax), %rdi                     #example_feautrier.cpp:123.2
  009de e8 fc ff ff ff  #       std::locale::~locale(std::locale *)
        call      _ZNSt6localeD1Ev                              #example_feautrier.cpp:123.2
  009e3 eb c1            jmp ..B1.192 # Prob 100%               #example_feautrier.cpp:123.2
                                # LOE
..B1.202:                       # Preds ..B1.121
                                # Execution count [1.25e-02]: Infreq
  009e5 bf fe ff ff ff   movl $.L_2__STRING.4, %edi             #example_feautrier.cpp:102.56
  009ea be fe ff ff ff   movl $.L_2__STRING.5, %esi             #example_feautrier.cpp:102.56
  009ef ba 42 00 00 00   movl $66, %edx                         #example_feautrier.cpp:102.56
  009f4 b9 00 00 00 00   movl $__$U9, %ecx                      #example_feautrier.cpp:102.56
  009f9 e8 fc ff ff ff  #       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #example_feautrier.cpp:102.56
                                # LOE
..B1.203:                       # Preds ..B1.84 ..B1.86 ..B1.88
                                # Execution count [9.00e-02]: Infreq
  009fe 45 33 c0         xorl %r8d, %r8d                        #example_feautrier.cpp:85.3
  00a01 e9 cd fb ff ff   jmp ..B1.100 # Prob 100%               #example_feautrier.cpp:85.3
                                # LOE rbx r8 r12 r13 r14 r15
..B1.206:                       # Preds ..B1.63 ..B1.65 ..B1.67
                                # Execution count [9.00e-02]: Infreq
  00a06 33 c9            xorl %ecx, %ecx                        #example_feautrier.cpp:79.3
  00a08 e9 7d fa ff ff   jmp ..B1.79 # Prob 100%                #example_feautrier.cpp:79.3
                                # LOE rcx rbx r12 r13 r14 r15
..B1.209:                       # Preds ..B1.35
                                # Execution count [5.28e-04]: Infreq
  00a0d bf 08 00 00 00   movl $8, %edi                          #example_feautrier.cpp:63.32
  00a12 e8 fc ff ff ff   call __cxa_allocate_exception          #example_feautrier.cpp:63.32
                                # LOE rax
..B1.210:                       # Preds ..B1.209
                                # Execution count [5.28e-04]: Infreq
  00a17 48 89 c7         movq %rax, %rdi                        #example_feautrier.cpp:63.32
  00a1a be 00 00 00 00   movl $_ZTISt9bad_alloc, %esi           #example_feautrier.cpp:63.32
  00a1f ba 00 00 00 00   movl $_ZNSt9bad_allocD1Ev, %edx        #example_feautrier.cpp:63.32
  00a24 48 c7 00 10 00 
        00 00            movq $_ZTVSt9bad_alloc+16, (%rax)      #example_feautrier.cpp:63.32
..___tag_value_main.137:
  00a2b e8 fc ff ff ff   call __cxa_throw                       #example_feautrier.cpp:63.32
..___tag_value_main.138:
                                # LOE
..___tag_value_main.12:
..B1.211:                       # Preds ..B1.210
                                # Execution count [0.00e+00]: Infreq
  00a30 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:63.32
                                # LOE
..B1.212:                       # Preds ..B1.211
                                # Execution count [0.00e+00]: Infreq
  00a34 48 8d 7c 24 30   lea 48(%rsp), %rdi                     #example_feautrier.cpp:63.32
..___tag_value_main.139:
  00a39 e8 fc ff ff ff  #       Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::~PlainObjectBase(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> *const)
        call      _ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev #example_feautrier.cpp:63.32
..___tag_value_main.140:
  00a3e e9 66 fe ff ff   jmp ..B1.165 # Prob 100%               #example_feautrier.cpp:63.32
                                # LOE
..B1.214:                       # Preds ..B1.34
                                # Execution count [5.01e-05]: Infreq
  00a43 bf fe ff ff ff   movl $.L_2__STRING.14, %edi            #example_feautrier.cpp:63.32
  00a48 be fe ff ff ff   movl $.L_2__STRING.15, %esi            #example_feautrier.cpp:63.32
  00a4d ba a1 00 00 00   movl $161, %edx                        #example_feautrier.cpp:63.32
  00a52 b9 00 00 00 00   movl $__$U2, %ecx                      #example_feautrier.cpp:63.32
  00a57 e8 fc ff ff ff  #       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #example_feautrier.cpp:63.32
                                # LOE
..___tag_value_main.7:
..B1.215:                       # Preds ..B1.30 ..B1.28 ..B1.25 ..B1.23 ..B1.21
                                #       ..B1.19 ..B1.16 ..B1.14 ..B1.12
                                # Execution count [0.00e+00]: Infreq
  00a5c 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
  00a60 e9 44 fe ff ff   jmp ..B1.165 # Prob 100%               #example_feautrier.cpp:43.25
                                # LOE
..___tag_value_main.3:
..B1.216:                       # Preds ..B1.11 ..B1.242 ..B1.8 ..B1.7
                                # Execution count [0.00e+00]: Infreq
  00a65 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
                                # LOE
..B1.217:                       # Preds ..B1.216
                                # Execution count [0.00e+00]: Infreq
  00a69 48 8d 84 24 88 
        00 00 00         lea 136(%rsp), %rax                    #example_feautrier.cpp:43.17
  00a71 48 c7 00 10 00 
        00 00            movq $_ZTVSt13basic_filebufIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.17
  00a78 48 8d 38         lea (%rax), %rdi                       #example_feautrier.cpp:43.25
..___tag_value_main.141:
  00a7b e8 fc ff ff ff  #       std::basic_filebuf<char, std::char_traits<char>>::close(std::basic_filebuf<char, std::char_traits<char>> *)
        call      _ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv #example_feautrier.cpp:43.25
..___tag_value_main.142:
                                # LOE
..B1.219:                       # Preds ..B1.217
                                # Execution count [0.00e+00]: Infreq
  00a80 48 8d bc 24 f0 
        00 00 00         lea 240(%rsp), %rdi                    #example_feautrier.cpp:43.25
  00a88 e8 fc ff ff ff  #       std::__basic_file<char>::~__basic_file(std::__basic_file<char> *)
        call      _ZNSt12__basic_fileIcED1Ev                    #example_feautrier.cpp:43.25
                                # LOE
..B1.220:                       # Preds ..B1.219
                                # Execution count [0.00e+00]: Infreq
  00a8d 48 8d 84 24 88 
        00 00 00         lea 136(%rsp), %rax                    #example_feautrier.cpp:43.17
  00a95 48 c7 00 10 00 
        00 00            movq $_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.17
  00a9c 48 8d 78 38      lea 56(%rax), %rdi                     #example_feautrier.cpp:43.25
  00aa0 e8 fc ff ff ff  #       std::locale::~locale(std::locale *)
        call      _ZNSt6localeD1Ev                              #example_feautrier.cpp:43.25
                                # LOE
..B1.225:                       # Preds ..B1.220 ..B1.243
                                # Execution count [0.00e+00]: Infreq
  00aa5 48 8b 05 04 00 
        00 00            movq 8+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rax #example_feautrier.cpp:43.25
  00aac 48 8b 15 0c 00 
        00 00            movq 16+_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE(%rip), %rdx #example_feautrier.cpp:43.25
  00ab3 48 8d 5c 24 78   lea 120(%rsp), %rbx                    #example_feautrier.cpp:43.17
  00ab8 48 89 03         movq %rax, (%rbx)                      #example_feautrier.cpp:43.17
  00abb 48 8b 48 e8      movq -24(%rax), %rcx                   #example_feautrier.cpp:43.25
  00abf 48 89 54 0c 78   movq %rdx, 120(%rsp,%rcx)              #example_feautrier.cpp:43.25
  00ac4 48 c7 43 08 00 
        00 00 00         movq $0, 8(%rbx)                       #example_feautrier.cpp:43.17
                                # LOE
..B1.229:                       # Preds ..B1.244 ..B1.225
                                # Execution count [0.00e+00]: Infreq
  00acc 48 8d 84 24 78 
        01 00 00         lea 376(%rsp), %rax                    #example_feautrier.cpp:43.17
  00ad4 48 c7 00 10 00 
        00 00            movq $_ZTVSt9basic_iosIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.17
  00adb 48 8d 38         lea (%rax), %rdi                       #example_feautrier.cpp:43.25
..___tag_value_main.143:
  00ade e8 fc ff ff ff  #       std::ios_base::~ios_base(std::ios_base *const)
        call      _ZNSt8ios_baseD2Ev                            #example_feautrier.cpp:43.25
..___tag_value_main.144:
  00ae3 e9 c9 fe ff ff   jmp ..B1.193 # Prob 100%               #example_feautrier.cpp:43.25
                                # LOE
..___tag_value_main.6:
..B1.231:                       # Preds ..B1.229
                                # Execution count [0.00e+00]: Infreq
  00ae8 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
                                # LOE
..B1.232:                       # Preds ..B1.239 ..B1.231
                                # Execution count [0.00e+00]: Infreq
  00aec 33 c0            xorl %eax, %eax                        #example_feautrier.cpp:43.25
  00aee 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:43.25
..___tag_value_main.145:
  00af2 e8 fc ff ff ff   call __cxa_call_unexpected             #example_feautrier.cpp:43.25
..___tag_value_main.146:
                                # LOE
..B1.233:                       # Preds ..B1.232
                                # Execution count [0.00e+00]: Infreq
  00af7 48 8b 3c 24      movq (%rsp), %rdi                      #example_feautrier.cpp:43.25
..___tag_value_main.147:
  00afb e8 fc ff ff ff   call _Unwind_Resume                    #example_feautrier.cpp:43.25
..___tag_value_main.148:
                                # LOE
..___tag_value_main.5:
..___tag_value_main.8:
..B1.237:                       # Preds ..B1.217
                                # Execution count [0.00e+00]: Infreq
  00b00 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
                                # LOE
..B1.238:                       # Preds ..B1.237
                                # Execution count [0.00e+00]: Infreq
  00b04 48 8d bc 24 f0 
        00 00 00         lea 240(%rsp), %rdi                    #example_feautrier.cpp:43.25
  00b0c e8 fc ff ff ff  #       std::__basic_file<char>::~__basic_file(std::__basic_file<char> *)
        call      _ZNSt12__basic_fileIcED1Ev                    #example_feautrier.cpp:43.25
                                # LOE
..B1.239:                       # Preds ..B1.238
                                # Execution count [0.00e+00]: Infreq
  00b11 48 8d 84 24 88 
        00 00 00         lea 136(%rsp), %rax                    #example_feautrier.cpp:43.17
  00b19 48 c7 00 10 00 
        00 00            movq $_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16, (%rax) #example_feautrier.cpp:43.17
  00b20 48 8d 78 38      lea 56(%rax), %rdi                     #example_feautrier.cpp:43.25
  00b24 e8 fc ff ff ff  #       std::locale::~locale(std::locale *)
        call      _ZNSt6localeD1Ev                              #example_feautrier.cpp:43.25
  00b29 eb c1            jmp ..B1.232 # Prob 100%               #example_feautrier.cpp:43.25
                                # LOE
..B1.242:                       # Preds ..B1.10
                                # Execution count [1.20e-01]: Infreq
  00b2b 48 8b 44 24 78   movq 120(%rsp), %rax                   #example_feautrier.cpp:43.17
  00b30 48 8b 50 e8      movq -24(%rax), %rdx                   #example_feautrier.cpp:43.25
  00b34 48 8d 7c 14 78   lea 120(%rsp,%rdx), %rdi               #example_feautrier.cpp:43.25
  00b39 8b 77 20         movl 32(%rdi), %esi                    #example_feautrier.cpp:43.25
  00b3c 83 ce 04         orl $4, %esi                           #example_feautrier.cpp:43.25
..___tag_value_main.149:
  00b3f e8 fc ff ff ff  #       std::basic_ios<char, std::char_traits<char>>::clear(std::basic_ios<char, std::char_traits<char>> *, std::ios_base::iostate)
        call      _ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate #example_feautrier.cpp:43.25
..___tag_value_main.150:
  00b44 e9 05 f6 ff ff   jmp ..B1.12 # Prob 100%                #example_feautrier.cpp:43.25
                                # LOE
..___tag_value_main.2:
..B1.243:                       # Preds ..B1.6
                                # Execution count [0.00e+00]: Infreq
  00b49 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
  00b4d e9 53 ff ff ff   jmp ..B1.225 # Prob 100%               #example_feautrier.cpp:43.25
                                # LOE
..___tag_value_main.1:
..B1.244:                       # Preds ..B1.5
                                # Execution count [0.00e+00]: Infreq
  00b52 48 89 04 24      movq %rax, (%rsp)                      #example_feautrier.cpp:43.25
  00b56 e9 71 ff ff ff   jmp ..B1.229 # Prob 100%               #example_feautrier.cpp:43.25
  00b5b 0f 1f 44 00 00   .align    16,0x90
                                # LOE
..___tag_value_main.4:
	.cfi_endproc
# mark_end;
	.type	main,@function
	.size	main,.-main
	.section .gcc_except_table, "a"
	.align 4
main$$LSDA:
	.byte	255
	.byte	0
	.uleb128	..___tag_value_main.155 - ..___tag_value_main.154
..___tag_value_main.154:
	.byte	1
	.uleb128	..___tag_value_main.153 - ..___tag_value_main.152
..___tag_value_main.152:
	.uleb128	..___tag_value_main.27 - 0
	.uleb128	..___tag_value_main.28 - ..___tag_value_main.27
	.byte	0
	.byte	0
	.uleb128	..___tag_value_main.29 - 0
	.uleb128	..___tag_value_main.30 - ..___tag_value_main.29
	.uleb128	..___tag_value_main.1 - 0
	.byte	0
	.uleb128	..___tag_value_main.31 - 0
	.uleb128	..___tag_value_main.32 - ..___tag_value_main.31
	.uleb128	..___tag_value_main.2 - 0
	.byte	0
	.uleb128	..___tag_value_main.33 - 0
	.uleb128	..___tag_value_main.38 - ..___tag_value_main.33
	.uleb128	..___tag_value_main.3 - 0
	.byte	0
	.uleb128	..___tag_value_main.39 - 0
	.uleb128	..___tag_value_main.56 - ..___tag_value_main.39
	.uleb128	..___tag_value_main.7 - 0
	.byte	0
	.uleb128	..___tag_value_main.57 - 0
	.uleb128	..___tag_value_main.100 - ..___tag_value_main.57
	.uleb128	..___tag_value_main.13 - 0
	.byte	0
	.uleb128	..___tag_value_main.101 - 0
	.uleb128	..___tag_value_main.102 - ..___tag_value_main.101
	.uleb128	..___tag_value_main.7 - 0
	.byte	0
	.uleb128	..___tag_value_main.103 - 0
	.uleb128	..___tag_value_main.104 - ..___tag_value_main.103
	.uleb128	..___tag_value_main.14 - 0
	.byte	3
	.uleb128	..___tag_value_main.105 - 0
	.uleb128	..___tag_value_main.106 - ..___tag_value_main.105
	.uleb128	..___tag_value_main.16 - 0
	.byte	1
	.uleb128	..___tag_value_main.107 - 0
	.uleb128	..___tag_value_main.108 - ..___tag_value_main.107
	.byte	0
	.byte	0
	.uleb128	..___tag_value_main.125 - 0
	.uleb128	..___tag_value_main.126 - ..___tag_value_main.125
	.uleb128	..___tag_value_main.9 - 0
	.byte	3
	.uleb128	..___tag_value_main.127 - 0
	.uleb128	..___tag_value_main.128 - ..___tag_value_main.127
	.uleb128	..___tag_value_main.11 - 0
	.byte	1
	.uleb128	..___tag_value_main.133 - 0
	.uleb128	..___tag_value_main.136 - ..___tag_value_main.133
	.byte	0
	.byte	0
	.uleb128	..___tag_value_main.137 - 0
	.uleb128	..___tag_value_main.138 - ..___tag_value_main.137
	.uleb128	..___tag_value_main.12 - 0
	.byte	0
	.uleb128	..___tag_value_main.141 - 0
	.uleb128	..___tag_value_main.142 - ..___tag_value_main.141
	.uleb128	..___tag_value_main.5 - 0
	.byte	3
	.uleb128	..___tag_value_main.143 - 0
	.uleb128	..___tag_value_main.144 - ..___tag_value_main.143
	.uleb128	..___tag_value_main.6 - 0
	.byte	1
	.uleb128	..___tag_value_main.149 - 0
	.uleb128	..___tag_value_main.150 - ..___tag_value_main.149
	.uleb128	..___tag_value_main.3 - 0
	.byte	0
..___tag_value_main.153:
	.byte	127
	.byte	0
	.byte	0
	.byte	125
	.long	0x00000000,0x00000000
..___tag_value_main.155:
	.byte	0
	.data
	.align 4
.2.249_2_kmpc_loc_struct_pack.525:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.249_2__kmpc_loc_pack.524
	.align 4
.2.249_2__kmpc_loc_pack.524:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	59
	.byte	51
	.byte	57
	.byte	59
	.byte	51
	.byte	57
	.byte	59
	.byte	59
	.space 3, 0x00 	# pad
	.align 4
.2.249_2_kmpc_loc_struct_pack.784:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.249_2__kmpc_loc_pack.783
	.align 4
.2.249_2__kmpc_loc_pack.783:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	59
	.byte	49
	.byte	50
	.byte	51
	.byte	59
	.byte	49
	.byte	50
	.byte	51
	.byte	59
	.byte	59
	.data
# -- End  main
	.section .text._ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE, "xaG",@progbits,_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE,comdat
..TXTST1:
# -- Begin  _ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE
	.section .text._ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE, "xaG",@progbits,_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE,comdat
# mark_begin;
       .align    16,0x90
	.weak _ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE
# --- Eigen::MapBase<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::internal::conditional<false, Eigen::InnerStride<1>, Eigen::OuterStride<-1>>::type>, 0>::checkSanity<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::internal::conditional<false, Eigen::InnerStride<1>, Eigen::OuterStride<-1>>::type>>(const Eigen::MapBase<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::internal::conditional<false, Eigen::InnerStride<1>, Eigen::OuterStride<-1>>::type>, 0> *, Eigen::internal::enable_if<true, void *>::type) const
_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE:
# parameter 1: %rdi
# parameter 2: %rsi
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE.156:
..L157:
                                                        #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/MapBase.h:198.5
  00000 c3               ret                                    #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/MapBase.h:198.6
  00001 0f 1f 84 00 00 
        00 00 00 0f 1f 
        80 00 00 00 00   .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE,@function
	.size	_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE,.-_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE
	.data
# -- End  _ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE
	.section .text._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev, "xaG",@progbits,_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev,comdat
..TXTST2:
# -- Begin  _ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev
	.section .text._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev, "xaG",@progbits,_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev,comdat
# mark_begin;
       .align    16,0x90
	.weak _ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev
# --- Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::~PlainObjectBase(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> *const)
_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev:
# parameter 1: %rdi
..B3.1:                         # Preds ..B3.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0xb, _ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev$$LSDA
..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.159:
..L160:
                                                        #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/PlainObjectBase.h:98.7
  00000 e9 fc ff ff ff  #       Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0>::~DenseStorage(Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0> *)
        jmp       _ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/PlainObjectBase.h:98.7
  00005 0f 1f 40 00 0f 
        1f 80 00 00 00 
        00               .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev,@function
	.size	_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev,.-_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev
	.section .gcc_except_table, "a"
	.align 4
_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev$$LSDA:
	.byte	255
	.byte	0
	.uleb128	..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.165 - ..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.164
..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.164:
	.byte	1
	.uleb128	..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.163 - ..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.162
..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.162:
..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.163:
	.long	0x00000000,0x00000000
..___tag_value__ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev.165:
	.data
# -- End  _ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEED2Ev
	.section .text._ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev, "xaG",@progbits,_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev,comdat
..TXTST3:
# -- Begin  _ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev
	.section .text._ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev, "xaG",@progbits,_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev,comdat
# mark_begin;
       .align    16,0x90
	.weak _ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev
# --- Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0>::~DenseStorage(Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0> *)
_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev:
# parameter 1: %rdi
..B4.1:                         # Preds ..B4.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev.166:
..L167:
                                                        #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/DenseStorage.h:391.39
  00000 48 8b 3f         movq (%rdi), %rdi                      #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/DenseStorage.h:391.41
  00003 e9 fc ff ff ff  #       free(void *)
        jmp       free                                          #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/DenseStorage.h:391.41
  00008 0f 1f 84 00 00 
        00 00 00         .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev,@function
	.size	_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev,.-_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev
	.data
# -- End  _ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev
	.section .text._ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev, "xaG",@progbits,_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev,comdat
..TXTST4:
# -- Begin  _ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev
	.section .text._ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev, "xaG",@progbits,_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev,comdat
# mark_begin;
       .align    16,0x90
	.weak _ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev
# --- Eigen::Matrix<double, -1, -1, 0, -1, -1>::~Matrix(Eigen::Matrix<double, -1, -1, 0, -1, -1> *)
_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev:
# parameter 1: %rdi
..B5.1:                         # Preds ..B5.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0xb, _ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev$$LSDA
..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.169:
..L170:
                                                        #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/Matrix.h:178.7
  00000 e9 fc ff ff ff  #       Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0>::~DenseStorage(Eigen::DenseStorage<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::Scalar, -1, -1, -1, 0> *)
        jmp       _ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EED1Ev #/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src/Eigen/src/Core/Matrix.h:178.7
  00005 0f 1f 40 00 0f 
        1f 80 00 00 00 
        00               .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev,@function
	.size	_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev,.-_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev
	.section .gcc_except_table, "a"
	.align 4
_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev$$LSDA:
	.byte	255
	.byte	0
	.uleb128	..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.175 - ..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.174
..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.174:
	.byte	1
	.uleb128	..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.173 - ..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.172
..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.172:
..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.173:
	.long	0x00000000,0x00000000
..___tag_value__ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev.175:
	.data
# -- End  _ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EED1Ev
	.text
# -- Begin  _Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl
# --- feautrier(long, double *, double *, long, double *, double *, double *, Eigen::Ref<Eigen::MatrixXd, 0, Eigen::internal::conditional<false, Eigen::InnerStride<1>, Eigen::OuterStride<-1>>::type> *, long)
_Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
# parameter 6: %r9
# parameter 7: 160 + %rsp
# parameter 8: 168 + %rsp
# parameter 9: 176 + %rsp
..B6.1:                         # Preds ..B6.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.176:
..L177:
                                                        #../src/feautrier.cpp:28.1
  00b60 41 54            pushq %r12                             #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
  00b62 41 55            pushq %r13                             #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
  00b64 41 56            pushq %r14                             #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
  00b66 41 57            pushq %r15                             #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
  00b68 53               pushq %rbx                             #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
  00b69 55               pushq %rbp                             #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
  00b6a 48 83 ec 68      subq $104, %rsp                        #../src/feautrier.cpp:28.1
	.cfi_def_cfa_offset 160
  00b6e 49 89 cc         movq %rcx, %r12                        #../src/feautrier.cpp:28.1
  00b71 49 89 fd         movq %rdi, %r13                        #../src/feautrier.cpp:28.1
  00b74 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:37.30
  00b7e 4c 89 4c 24 58   movq %r9, 88(%rsp)                     #../src/feautrier.cpp:28.1[spill]
  00b83 66 0f ef c0      pxor %xmm0, %xmm0                      #../src/feautrier.cpp:32.2
  00b87 4c 89 44 24 50   movq %r8, 80(%rsp)                     #../src/feautrier.cpp:28.1[spill]
  00b8c 48 89 54 24 48   movq %rdx, 72(%rsp)                    #../src/feautrier.cpp:28.1[spill]
  00b91 4b 8d 5c 25 00   lea (%r13,%r12), %rbx                  #../src/feautrier.cpp:30.20
  00b96 48 89 74 24 20   movq %rsi, 32(%rsp)                    #../src/feautrier.cpp:28.1[spill]
  00b9b f2 0f 11 44 24 
        28               movsd %xmm0, 40(%rsp)                  #../src/feautrier.cpp:32.2[spill]
  00ba1 f2 0f 11 44 24 
        30               movsd %xmm0, 48(%rsp)                  #../src/feautrier.cpp:33.3[spill]
  00ba7 f2 0f 11 44 24 
        38               movsd %xmm0, 56(%rsp)                  #../src/feautrier.cpp:34.3[spill]
  00bad f2 0f 11 44 24 
        40               movsd %xmm0, 64(%rsp)                  #../src/feautrier.cpp:35.2[spill]
  00bb3 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:37.30
  00bb6 73 2e            jae ..B6.6 # Prob 50%                  #../src/feautrier.cpp:37.30
                                # LOE rbx r12 r13
..B6.2:                         # Preds ..B6.1 ..B6.6
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.191:
  00bb8 e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:37.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.192:
                                # LOE rbx r12 r13
..B6.3:                         # Preds ..B6.2
                                # Execution count [5.25e-01]
  00bbd 48 8d 2c dd 00 
        00 00 00         lea (,%rbx,8), %rbp                    #../src/feautrier.cpp:37.30
  00bc5 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:37.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.193:
  00bc8 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:37.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.194:
                                # LOE rax rbx rbp r12 r13
..B6.201:                       # Preds ..B6.3
                                # Execution count [5.25e-01]
  00bcd 49 89 c7         movq %rax, %r15                        #../src/feautrier.cpp:37.30
                                # LOE rbx rbp r12 r13 r15
..B6.4:                         # Preds ..B6.201
                                # Execution count [5.25e-01]
  00bd0 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:38.30
  00bda 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:38.30
  00bdd 72 2e            jb ..B6.9 # Prob 50%                   #../src/feautrier.cpp:38.30
                                # LOE rbx rbp r12 r13 r15
..B6.5:                         # Preds ..B6.4
                                # Execution count [2.63e-01]
  00bdf 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:38.30
  00be2 78 29            js ..B6.9 # Prob 5%                    #../src/feautrier.cpp:38.30
  00be4 eb 4d            jmp ..B6.13 # Prob 100%                #../src/feautrier.cpp:38.30
                                # LOE rbx rbp r12 r13 r15
..B6.6:                         # Preds ..B6.1
                                # Execution count [5.00e-01]
  00be6 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:37.30
  00be9 78 cd            js ..B6.2 # Prob 5%                    #../src/feautrier.cpp:37.30
                                # LOE rbx r12 r13
..B6.7:                         # Preds ..B6.6
                                # Execution count [4.75e-01]
  00beb 48 8d 2c dd 00 
        00 00 00         lea (,%rbx,8), %rbp                    #../src/feautrier.cpp:37.30
  00bf3 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:37.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.195:
  00bf6 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:37.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.196:
                                # LOE rax rbx rbp r12 r13
..B6.202:                       # Preds ..B6.7
                                # Execution count [4.75e-01]
  00bfb 49 89 c7         movq %rax, %r15                        #../src/feautrier.cpp:37.30
                                # LOE rbx rbp r12 r13 r15
..B6.8:                         # Preds ..B6.202
                                # Execution count [4.75e-01]
  00bfe 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:38.30
  00c08 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:38.30
  00c0b 73 26            jae ..B6.13 # Prob 50%                 #../src/feautrier.cpp:38.30
                                # LOE rbx rbp r12 r13 r15
..B6.9:                         # Preds ..B6.8 ..B6.5 ..B6.4
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.197:
  00c0d e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:38.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.198:
                                # LOE rbx rbp r12 r13 r15
..B6.10:                        # Preds ..B6.9
                                # Execution count [5.25e-01]
  00c12 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:38.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.199:
  00c15 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:38.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.200:
                                # LOE rax rbx rbp r12 r13 r15
..B6.203:                       # Preds ..B6.10
                                # Execution count [5.25e-01]
  00c1a 49 89 c6         movq %rax, %r14                        #../src/feautrier.cpp:38.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.11:                        # Preds ..B6.203
                                # Execution count [5.25e-01]
  00c1d 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:39.30
  00c27 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:39.30
  00c2a 72 21            jb ..B6.15 # Prob 50%                  #../src/feautrier.cpp:39.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.12:                        # Preds ..B6.11
                                # Execution count [2.63e-01]
  00c2c 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:39.30
  00c2f 78 1c            js ..B6.15 # Prob 5%                   #../src/feautrier.cpp:39.30
  00c31 eb 42            jmp ..B6.19 # Prob 100%                #../src/feautrier.cpp:39.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.13:                        # Preds ..B6.8 ..B6.5
                                # Execution count [4.75e-01]
  00c33 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:38.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.201:
  00c36 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:38.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.202:
                                # LOE rax rbx rbp r12 r13 r15
..B6.204:                       # Preds ..B6.13
                                # Execution count [4.75e-01]
  00c3b 49 89 c6         movq %rax, %r14                        #../src/feautrier.cpp:38.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.14:                        # Preds ..B6.204
                                # Execution count [4.75e-01]
  00c3e 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:39.30
  00c48 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:39.30
  00c4b 73 28            jae ..B6.19 # Prob 50%                 #../src/feautrier.cpp:39.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.15:                        # Preds ..B6.14 ..B6.12 ..B6.11
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.203:
  00c4d e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:39.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.204:
                                # LOE rbx rbp r12 r13 r14 r15
..B6.16:                        # Preds ..B6.15
                                # Execution count [5.25e-01]
  00c52 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:39.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.205:
  00c55 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:39.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.206:
                                # LOE rax rbx rbp r12 r13 r14 r15
..B6.205:                       # Preds ..B6.16
                                # Execution count [5.25e-01]
  00c5a 48 89 44 24 60   movq %rax, 96(%rsp)                    #../src/feautrier.cpp:39.30[spill]
                                # LOE rbx rbp r12 r13 r14 r15
..B6.17:                        # Preds ..B6.205
                                # Execution count [5.25e-01]
  00c5f 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:40.30
  00c69 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:40.30
  00c6c 72 23            jb ..B6.21 # Prob 50%                  #../src/feautrier.cpp:40.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.18:                        # Preds ..B6.17
                                # Execution count [2.63e-01]
  00c6e 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:40.30
  00c71 78 1e            js ..B6.21 # Prob 5%                   #../src/feautrier.cpp:40.30
  00c73 eb 21            jmp ..B6.22 # Prob 100%                #../src/feautrier.cpp:40.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.19:                        # Preds ..B6.14 ..B6.12
                                # Execution count [4.75e-01]
  00c75 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:39.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.207:
  00c78 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:39.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.208:
                                # LOE rax rbx rbp r12 r13 r14 r15
..B6.206:                       # Preds ..B6.19
                                # Execution count [4.75e-01]
  00c7d 48 89 44 24 60   movq %rax, 96(%rsp)                    #../src/feautrier.cpp:39.30[spill]
                                # LOE rbx rbp r12 r13 r14 r15
..B6.20:                        # Preds ..B6.206
                                # Execution count [4.75e-01]
  00c82 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:40.30
  00c8c 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:40.30
  00c8f 73 05            jae ..B6.22 # Prob 50%                 #../src/feautrier.cpp:40.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.21:                        # Preds ..B6.20 ..B6.18 ..B6.17
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.209:
  00c91 e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:40.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.210:
                                # LOE rbx rbp r12 r13 r14 r15
..B6.22:                        # Preds ..B6.20 ..B6.18 ..B6.21
                                # Execution count [1.00e+00]
  00c96 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:40.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.211:
  00c99 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:40.30
..___tag_value__Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl.212:
                                # LOE rax rbx r12 r13 r14 r15
..B6.207:                       # Preds ..B6.22
                                # Execution count [1.00e+00]
  00c9e 48 89 c5         movq %rax, %rbp                        #../src/feautrier.cpp:40.30
                                # LOE rbx rbp r12 r13 r14 r15
..B6.23:                        # Preds ..B6.207
                                # Execution count [1.00e+00]
  00ca1 4c 8b 84 24 a0 
        00 00 00         movq 160(%rsp), %r8                    #../src/feautrier.cpp:28.1
  00ca9 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:49.15
  00cac 0f 8e 36 0f 00 
        00               jle ..B6.194 # Prob 16%                #../src/feautrier.cpp:49.15
                                # LOE rbx rbp r8 r12 r13 r14 r15
..B6.24:                        # Preds ..B6.23
                                # Execution count [8.40e-01]
  00cb2 f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:51.17
  00cba 4a 8d 0c e5 00 
        00 00 00         lea (,%r12,8), %rcx                    #../src/feautrier.cpp:51.5
  00cc2 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:49.28
  00cc5 7e 39            jle ..B6.26 # Prob 16%                 #../src/feautrier.cpp:49.28
                                # LOE rcx rbx rbp r8 r12 r13 r14 r15 xmm1
..B6.25:                        # Preds ..B6.24
                                # Execution count [7.06e-01]
  00cc7 48 8b 44 24 58   movq 88(%rsp), %rax                    #../src/feautrier.cpp:51.24[spill]
  00ccc 0f 28 e9         movaps %xmm1, %xmm5                    #../src/feautrier.cpp:51.37
  00ccf 48 8b 54 24 48   movq 72(%rsp), %rdx                    #../src/feautrier.cpp:51.37[spill]
  00cd4 f2 0f 10 10      movsd (%rax), %xmm2                    #../src/feautrier.cpp:51.24
  00cd8 f2 0f 10 22      movsd (%rdx), %xmm4                    #../src/feautrier.cpp:51.37
  00cdc 0f 28 c2         movaps %xmm2, %xmm0                    #../src/feautrier.cpp:51.37
  00cdf f2 0f 58 c4      addsd %xmm4, %xmm0                     #../src/feautrier.cpp:51.37
  00ce3 f2 0f 5e e8      divsd %xmm0, %xmm5                     #../src/feautrier.cpp:51.37
  00ce7 0f 28 dd         movaps %xmm5, %xmm3                    #../src/feautrier.cpp:51.50
  00cea f2 0f 5e da      divsd %xmm2, %xmm3                     #../src/feautrier.cpp:51.50
  00cee f2 0f 5e ec      divsd %xmm4, %xmm5                     #../src/feautrier.cpp:52.50
  00cf2 f2 41 0f 11 5c 
        0f f8            movsd %xmm3, -8(%r15,%rcx)             #../src/feautrier.cpp:51.5
  00cf9 f2 41 0f 11 6c 
        0e f8            movsd %xmm5, -8(%r14,%rcx)             #../src/feautrier.cpp:52.5
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1
..B6.26:                        # Preds ..B6.24 ..B6.25
                                # Execution count [8.40e-01]
  00d00 48 8b 54 24 58   movq 88(%rsp), %rdx                    #../src/feautrier.cpp:59.32[spill]
  00d05 0f 28 d9         movaps %xmm1, %xmm3                    #../src/feautrier.cpp:59.16
  00d08 49 8d 44 24 ff   lea -1(%r12), %rax                     #../src/feautrier.cpp:64.17
  00d0d 48 89 44 24 18   movq %rax, 24(%rsp)                    #../src/feautrier.cpp:64.17[spill]
  00d12 66 0f ef d2      pxor %xmm2, %xmm2                      #../src/feautrier.cpp:58.5
  00d16 f2 41 0f 11 17   movsd %xmm2, (%r15)                    #../src/feautrier.cpp:58.5
  00d1b 4a 8d 14 e2      lea (%rdx,%r12,8), %rdx                #../src/feautrier.cpp:59.32
  00d1f f2 0f 10 42 f8   movsd -8(%rdx), %xmm0                  #../src/feautrier.cpp:59.16
  00d24 f2 0f 5e d8      divsd %xmm0, %xmm3                     #../src/feautrier.cpp:59.16
  00d28 0f 28 e3         movaps %xmm3, %xmm4                    #../src/feautrier.cpp:59.32
  00d2b f2 0f 5e e0      divsd %xmm0, %xmm4                     #../src/feautrier.cpp:59.32
  00d2f f2 0f 10 15 fc 
        ff ff ff         movsd .L_2il0floatpacket.62(%rip), %xmm2 #../src/feautrier.cpp:61.5
  00d37 f2 41 0f 11 26   movsd %xmm4, (%r14)                    #../src/feautrier.cpp:59.5
  00d3c f2 0f 58 da      addsd %xmm2, %xmm3                     #../src/feautrier.cpp:61.27
  00d40 f2 0f 11 5c 24 
        30               movsd %xmm3, 48(%rsp)                  #../src/feautrier.cpp:61.27[spill]
  00d46 f2 0f 58 e3      addsd %xmm3, %xmm4                     #../src/feautrier.cpp:61.65
  00d4a f2 0f 11 64 24 
        28               movsd %xmm4, 40(%rsp)                  #../src/feautrier.cpp:61.65[spill]
  00d50 48 83 f8 01      cmpq $1, %rax                          #../src/feautrier.cpp:64.31
  00d54 0f 8e 6d 04 00 
        00               jle ..B6.52 # Prob 50%                 #../src/feautrier.cpp:64.31
                                # LOE rdx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.27:                        # Preds ..B6.26
                                # Execution count [7.56e-01]
  00d5a 49 8d 4c 24 fe   lea -2(%r12), %rcx                     #../src/feautrier.cpp:64.5
  00d5f 48 83 f9 08      cmpq $8, %rcx                          #../src/feautrier.cpp:64.5
  00d63 0f 8c 6c 0e 00 
        00               jl ..B6.192 # Prob 10%                 #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.28:                        # Preds ..B6.27
                                # Execution count [7.56e-01]
  00d69 49 8d 46 08      lea 8(%r14), %rax                      #../src/feautrier.cpp:67.7
  00d6d 48 83 e0 0f      andq $15, %rax                         #../src/feautrier.cpp:64.5
  00d71 74 11            je ..B6.31 # Prob 50%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.29:                        # Preds ..B6.28
                                # Execution count [7.56e-01]
  00d73 48 a9 07 00 00 
        00               testq $7, %rax                         #../src/feautrier.cpp:64.5
  00d79 0f 85 56 0e 00 
        00               jne ..B6.192 # Prob 10%                #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.30:                        # Preds ..B6.29
                                # Execution count [3.78e-01]
  00d7f b8 01 00 00 00   movl $1, %eax                          #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.31:                        # Preds ..B6.30 ..B6.28
                                # Execution count [7.56e-01]
  00d84 48 8d 70 08      lea 8(%rax), %rsi                      #../src/feautrier.cpp:64.5
  00d88 48 3b ce         cmpq %rsi, %rcx                        #../src/feautrier.cpp:64.5
  00d8b 0f 8c 44 0e 00 
        00               jl ..B6.192 # Prob 10%                 #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.32:                        # Preds ..B6.31
                                # Execution count [8.40e-01]
  00d91 48 89 ce         movq %rcx, %rsi                        #../src/feautrier.cpp:64.5
  00d94 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:64.5
  00d97 48 2b f0         subq %rax, %rsi                        #../src/feautrier.cpp:64.5
  00d9a 45 33 c9         xorl %r9d, %r9d                        #../src/feautrier.cpp:66.7
  00d9d 48 83 e6 07      andq $7, %rsi                          #../src/feautrier.cpp:64.5
  00da1 48 f7 de         negq %rsi                              #../src/feautrier.cpp:64.5
  00da4 48 03 f1         addq %rcx, %rsi                        #../src/feautrier.cpp:64.5
  00da7 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:64.5
  00daa 76 40            jbe ..B6.36 # Prob 9%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.34:                        # Preds ..B6.32 ..B6.34
                                # Execution count [4.20e+00]
  00dac f2 42 0f 10 5c 
        ca f0            movsd -16(%rdx,%r9,8), %xmm3           #../src/feautrier.cpp:66.57
  00db3 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:66.41
  00db6 f2 42 0f 10 6c 
        ca e8            movsd -24(%rdx,%r9,8), %xmm5           #../src/feautrier.cpp:66.41
  00dbd 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:66.41
  00dc0 49 ff c9         decq %r9                               #../src/feautrier.cpp:64.5
  00dc3 f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:66.41
  00dc7 f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:66.41
  00dcb 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:66.57
  00dce f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:66.57
  00dd2 f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:67.57
  00dd6 f2 43 0f 11 64 
        d7 08            movsd %xmm4, 8(%r15,%r10,8)            #../src/feautrier.cpp:66.7
  00ddd f2 43 0f 11 74 
        d6 08            movsd %xmm6, 8(%r14,%r10,8)            #../src/feautrier.cpp:67.7
  00de4 49 ff c2         incq %r10                              #../src/feautrier.cpp:64.5
  00de7 4c 3b d0         cmpq %rax, %r10                        #../src/feautrier.cpp:64.5
  00dea 72 c0            jb ..B6.34 # Prob 82%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.36:                        # Preds ..B6.34 ..B6.32
                                # Execution count [0.00e+00]
  00dec 0f 10 05 fc ff 
        ff ff            movups .L_2il0floatpacket.61(%rip), %xmm0 #../src/feautrier.cpp:66.19
  00df3 4c 8d 0c c5 00 
        00 00 00         lea (,%rax,8), %r9                     #../src/feautrier.cpp:64.5
  00dfb 49 f7 d9         negq %r9                               #../src/feautrier.cpp:64.5
  00dfe 4d 8d 54 c7 08   lea 8(%r15,%rax,8), %r10               #../src/feautrier.cpp:66.7
  00e03 4c 89 0c 24      movq %r9, (%rsp)                       #../src/feautrier.cpp:64.5[spill]
  00e07 49 f7 c2 0f 00 
        00 00            testq $15, %r10                        #../src/feautrier.cpp:64.5
  00e0e 0f 84 52 01 00 
        00               je ..B6.40 # Prob 60%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.37:                        # Preds ..B6.36
                                # Execution count [7.56e-01]
  00e14 4c 8b 4c 24 58   movq 88(%rsp), %r9                     #../src/feautrier.cpp:58.5[spill]
  00e19 4c 89 64 24 10   movq %r12, 16(%rsp)                    #../src/feautrier.cpp:58.5[spill]
  00e1e 4c 89 6c 24 08   movq %r13, 8(%rsp)                     #../src/feautrier.cpp:58.5[spill]
  00e23 4f 8d 5c e1 e8   lea -24(%r9,%r12,8), %r11              #../src/feautrier.cpp:58.5
  00e28 4f 8d 54 e1 d8   lea -40(%r9,%r12,8), %r10              #../src/feautrier.cpp:58.5
  00e2d 4b 8d 7c e1 c8   lea -56(%r9,%r12,8), %rdi              #../src/feautrier.cpp:58.5
  00e32 4f 8d 4c e1 b8   lea -72(%r9,%r12,8), %r9               #../src/feautrier.cpp:58.5
  00e37 4c 8b 24 24      movq (%rsp), %r12                      #../src/feautrier.cpp:58.5[spill]
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B6.38:                        # Preds ..B6.37 ..B6.38
                                # Execution count [4.20e+00]
  00e3b 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:66.41
  00e3e 4f 8d 2c 23      lea (%r11,%r12), %r13                  #../src/feautrier.cpp:66.41
  00e42 f2 41 0f 10 65 
        08               movsd 8(%r13), %xmm4                   #../src/feautrier.cpp:66.26
  00e48 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:66.41
  00e4c 66 41 0f 16 65 
        00               movhpd (%r13), %xmm4                   #../src/feautrier.cpp:66.26
  00e52 f2 41 0f 10 75 
        00               movsd (%r13), %xmm6                    #../src/feautrier.cpp:66.26
  00e58 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:66.41
  00e5b 66 41 0f 16 75 
        f8               movhpd -8(%r13), %xmm6                 #../src/feautrier.cpp:66.26
  00e61 4f 8d 2c 22      lea (%r10,%r12), %r13                  #../src/feautrier.cpp:66.41
  00e65 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:66.41
  00e69 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:66.41
  00e6d 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:66.57
  00e70 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:66.57
  00e74 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:67.57
  00e78 41 0f 11 6c c7 
        08               movups %xmm5, 8(%r15,%rax,8)           #../src/feautrier.cpp:66.7
  00e7e 41 0f 11 7c c6 
        08               movups %xmm7, 8(%r14,%rax,8)           #../src/feautrier.cpp:67.7
  00e84 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:66.41
  00e87 f2 45 0f 10 4d 
        08               movsd 8(%r13), %xmm9                   #../src/feautrier.cpp:66.26
  00e8d 66 45 0f 16 4d 
        00               movhpd (%r13), %xmm9                   #../src/feautrier.cpp:66.26
  00e93 f2 45 0f 10 5d 
        00               movsd (%r13), %xmm11                   #../src/feautrier.cpp:66.26
  00e99 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:66.41
  00e9d 66 45 0f 16 5d 
        f8               movhpd -8(%r13), %xmm11                #../src/feautrier.cpp:66.26
  00ea3 4e 8d 2c 27      lea (%rdi,%r12), %r13                  #../src/feautrier.cpp:66.41
  00ea7 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:66.41
  00eac 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:66.41
  00eb1 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:66.57
  00eb5 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:66.57
  00eba 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:67.57
  00ebf 45 0f 11 54 c7 
        18               movups %xmm10, 24(%r15,%rax,8)         #../src/feautrier.cpp:66.7
  00ec5 45 0f 11 64 c6 
        18               movups %xmm12, 24(%r14,%rax,8)         #../src/feautrier.cpp:67.7
  00ecb 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:66.41
  00ecf f2 45 0f 10 75 
        08               movsd 8(%r13), %xmm14                  #../src/feautrier.cpp:66.26
  00ed5 66 45 0f 16 75 
        00               movhpd (%r13), %xmm14                  #../src/feautrier.cpp:66.26
  00edb f2 41 0f 10 5d 
        00               movsd (%r13), %xmm3                    #../src/feautrier.cpp:66.26
  00ee1 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:66.41
  00ee5 66 41 0f 16 5d 
        f8               movhpd -8(%r13), %xmm3                 #../src/feautrier.cpp:66.26
  00eeb 4f 8d 2c 21      lea (%r9,%r12), %r13                   #../src/feautrier.cpp:66.41
  00eef 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:66.41
  00ef4 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:66.41
  00ef9 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:66.57
  00efd 49 83 c4 c0      addq $-64, %r12                        #../src/feautrier.cpp:64.5
  00f01 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:66.57
  00f06 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:67.57
  00f0a 45 0f 11 7c c7 
        28               movups %xmm15, 40(%r15,%rax,8)         #../src/feautrier.cpp:66.7
  00f10 41 0f 11 64 c6 
        28               movups %xmm4, 40(%r14,%rax,8)          #../src/feautrier.cpp:67.7
  00f16 f2 41 0f 10 75 
        08               movsd 8(%r13), %xmm6                   #../src/feautrier.cpp:66.26
  00f1c 66 41 0f 16 75 
        00               movhpd (%r13), %xmm6                   #../src/feautrier.cpp:66.26
  00f22 f2 45 0f 10 45 
        00               movsd (%r13), %xmm8                    #../src/feautrier.cpp:66.26
  00f28 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:66.41
  00f2b 66 45 0f 16 45 
        f8               movhpd -8(%r13), %xmm8                 #../src/feautrier.cpp:66.26
  00f31 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:66.41
  00f36 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:66.41
  00f3b 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:66.57
  00f3f 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:66.57
  00f43 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:67.57
  00f48 41 0f 11 7c c7 
        38               movups %xmm7, 56(%r15,%rax,8)          #../src/feautrier.cpp:66.7
  00f4e 45 0f 11 4c c6 
        38               movups %xmm9, 56(%r14,%rax,8)          #../src/feautrier.cpp:67.7
  00f54 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:64.5
  00f58 48 3b c6         cmpq %rsi, %rax                        #../src/feautrier.cpp:64.5
  00f5b 0f 82 da fe ff 
        ff               jb ..B6.38 # Prob 82%                  #../src/feautrier.cpp:64.5
  00f61 e9 4d 01 00 00   jmp ..B6.42 # Prob 100%                #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B6.40:                        # Preds ..B6.36
                                # Execution count [7.56e-01]
  00f66 4c 8b 5c 24 58   movq 88(%rsp), %r11                    #../src/feautrier.cpp:58.5[spill]
  00f6b 4c 89 64 24 10   movq %r12, 16(%rsp)                    #../src/feautrier.cpp:58.5[spill]
  00f70 4c 89 6c 24 08   movq %r13, 8(%rsp)                     #../src/feautrier.cpp:58.5[spill]
  00f75 4f 8d 54 e3 e8   lea -24(%r11,%r12,8), %r10             #../src/feautrier.cpp:58.5
  00f7a 4f 8d 4c e3 d8   lea -40(%r11,%r12,8), %r9              #../src/feautrier.cpp:58.5
  00f7f 4b 8d 7c e3 c8   lea -56(%r11,%r12,8), %rdi             #../src/feautrier.cpp:58.5
  00f84 4f 8d 5c e3 b8   lea -72(%r11,%r12,8), %r11             #../src/feautrier.cpp:58.5
  00f89 4c 8b 24 24      movq (%rsp), %r12                      #../src/feautrier.cpp:58.5[spill]
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B6.41:                        # Preds ..B6.40 ..B6.41
                                # Execution count [4.20e+00]
  00f8d 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:66.41
  00f90 4f 8d 2c 22      lea (%r10,%r12), %r13                  #../src/feautrier.cpp:66.41
  00f94 f2 41 0f 10 65 
        08               movsd 8(%r13), %xmm4                   #../src/feautrier.cpp:66.26
  00f9a 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:66.41
  00f9e 66 41 0f 16 65 
        00               movhpd (%r13), %xmm4                   #../src/feautrier.cpp:66.26
  00fa4 f2 41 0f 10 75 
        00               movsd (%r13), %xmm6                    #../src/feautrier.cpp:66.26
  00faa 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:66.41
  00fad 66 41 0f 16 75 
        f8               movhpd -8(%r13), %xmm6                 #../src/feautrier.cpp:66.26
  00fb3 4f 8d 2c 21      lea (%r9,%r12), %r13                   #../src/feautrier.cpp:66.41
  00fb7 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:66.41
  00fbb 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:66.41
  00fbf 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:66.57
  00fc2 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:66.57
  00fc6 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:67.57
  00fca 41 0f 11 6c c7 
        08               movups %xmm5, 8(%r15,%rax,8)           #../src/feautrier.cpp:66.7
  00fd0 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:66.41
  00fd3 41 0f 11 7c c6 
        08               movups %xmm7, 8(%r14,%rax,8)           #../src/feautrier.cpp:67.7
  00fd9 f2 45 0f 10 4d 
        08               movsd 8(%r13), %xmm9                   #../src/feautrier.cpp:66.26
  00fdf 66 45 0f 16 4d 
        00               movhpd (%r13), %xmm9                   #../src/feautrier.cpp:66.26
  00fe5 f2 45 0f 10 5d 
        00               movsd (%r13), %xmm11                   #../src/feautrier.cpp:66.26
  00feb 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:66.41
  00fef 66 45 0f 16 5d 
        f8               movhpd -8(%r13), %xmm11                #../src/feautrier.cpp:66.26
  00ff5 4e 8d 2c 27      lea (%rdi,%r12), %r13                  #../src/feautrier.cpp:66.41
  00ff9 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:66.41
  00ffe 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:66.41
  01003 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:66.57
  01007 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:66.57
  0100c 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:67.57
  01011 45 0f 11 54 c7 
        18               movups %xmm10, 24(%r15,%rax,8)         #../src/feautrier.cpp:66.7
  01017 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:66.41
  0101b 45 0f 11 64 c6 
        18               movups %xmm12, 24(%r14,%rax,8)         #../src/feautrier.cpp:67.7
  01021 f2 45 0f 10 75 
        08               movsd 8(%r13), %xmm14                  #../src/feautrier.cpp:66.26
  01027 66 45 0f 16 75 
        00               movhpd (%r13), %xmm14                  #../src/feautrier.cpp:66.26
  0102d f2 41 0f 10 5d 
        00               movsd (%r13), %xmm3                    #../src/feautrier.cpp:66.26
  01033 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:66.41
  01037 66 41 0f 16 5d 
        f8               movhpd -8(%r13), %xmm3                 #../src/feautrier.cpp:66.26
  0103d 4f 8d 2c 23      lea (%r11,%r12), %r13                  #../src/feautrier.cpp:66.41
  01041 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:66.41
  01046 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:66.41
  0104b 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:66.57
  0104f 49 83 c4 c0      addq $-64, %r12                        #../src/feautrier.cpp:64.5
  01053 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:66.57
  01058 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:67.57
  0105c 45 0f 11 7c c7 
        28               movups %xmm15, 40(%r15,%rax,8)         #../src/feautrier.cpp:66.7
  01062 41 0f 11 64 c6 
        28               movups %xmm4, 40(%r14,%rax,8)          #../src/feautrier.cpp:67.7
  01068 f2 41 0f 10 75 
        08               movsd 8(%r13), %xmm6                   #../src/feautrier.cpp:66.26
  0106e 66 41 0f 16 75 
        00               movhpd (%r13), %xmm6                   #../src/feautrier.cpp:66.26
  01074 f2 45 0f 10 45 
        00               movsd (%r13), %xmm8                    #../src/feautrier.cpp:66.26
  0107a 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:66.41
  0107d 66 45 0f 16 45 
        f8               movhpd -8(%r13), %xmm8                 #../src/feautrier.cpp:66.26
  01083 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:66.41
  01088 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:66.41
  0108d 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:66.57
  01091 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:66.57
  01095 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:67.57
  0109a 41 0f 11 7c c7 
        38               movups %xmm7, 56(%r15,%rax,8)          #../src/feautrier.cpp:66.7
  010a0 45 0f 11 4c c6 
        38               movups %xmm9, 56(%r14,%rax,8)          #../src/feautrier.cpp:67.7
  010a6 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:64.5
  010aa 48 3b c6         cmpq %rsi, %rax                        #../src/feautrier.cpp:64.5
  010ad 0f 82 da fe ff 
        ff               jb ..B6.41 # Prob 82%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B6.42:                        # Preds ..B6.38 ..B6.41
                                # Execution count [7.56e-01]
  010b3 4c 8b 6c 24 08   movq 8(%rsp), %r13                     #[spill]
  010b8 4c 8b 64 24 10   movq 16(%rsp), %r12                    #[spill]
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.43:                        # Preds ..B6.42
                                # Execution count [7.56e-01]
  010bd 48 8d 46 01      lea 1(%rsi), %rax                      #../src/feautrier.cpp:64.5
  010c1 48 3b c1         cmpq %rcx, %rax                        #../src/feautrier.cpp:64.5
  010c4 0f 87 07 01 00 
        00               ja ..B6.53 # Prob 50%                  #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.44:                        # Preds ..B6.43
                                # Execution count [7.56e-01]
  010ca 48 2b ce         subq %rsi, %rcx                        #../src/feautrier.cpp:64.5
  010cd 48 83 f9 02      cmpq $2, %rcx                          #../src/feautrier.cpp:64.5
  010d1 0f 8c 0a 0b 00 
        00               jl ..B6.193 # Prob 10%                 #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.45:                        # Preds ..B6.44
                                # Execution count [7.56e-01]
  010d7 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:64.5
  010da 48 8d 04 f5 00 
        00 00 00         lea (,%rsi,8), %rax                    #../src/feautrier.cpp:66.41
  010e2 48 f7 d8         negq %rax                              #../src/feautrier.cpp:66.41
  010e5 48 89 cf         movq %rcx, %rdi                        #../src/feautrier.cpp:64.5
  010e8 48 83 e7 fe      andq $-2, %rdi                         #../src/feautrier.cpp:64.5
  010ec 4c 89 6c 24 08   movq %r13, 8(%rsp)                     #../src/feautrier.cpp:58.5[spill]
  010f1 4d 8d 0c f7      lea (%r15,%rsi,8), %r9                 #../src/feautrier.cpp:66.7
  010f5 4c 89 64 24 10   movq %r12, 16(%rsp)                    #../src/feautrier.cpp:58.5[spill]
  010fa 4d 8d 14 f6      lea (%r14,%rsi,8), %r10                #../src/feautrier.cpp:67.7
  010fe 4d 89 dd         movq %r11, %r13                        #../src/feautrier.cpp:58.5
  01101 48 8d 44 02 e8   lea -24(%rdx,%rax), %rax               #../src/feautrier.cpp:58.5
  01106 0f 1f 00 0f 1f 
        80 00 00 00 00   .align    16,0x90
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r13 r14 r15 xmm0 xmm1 xmm2
..B6.46:                        # Preds ..B6.46 ..B6.45
                                # Execution count [4.20e+00]
  01110 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:66.41
  01113 4e 8d 24 28      lea (%rax,%r13), %r12                  #../src/feautrier.cpp:66.41
  01117 f2 41 0f 10 64 
        24 08            movsd 8(%r12), %xmm4                   #../src/feautrier.cpp:66.26
  0111e 49 83 c5 f0      addq $-16, %r13                        #../src/feautrier.cpp:64.5
  01122 66 41 0f 16 24 
        24               movhpd (%r12), %xmm4                   #../src/feautrier.cpp:66.26
  01128 f2 41 0f 10 34 
        24               movsd (%r12), %xmm6                    #../src/feautrier.cpp:66.26
  0112e 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:66.41
  01131 66 41 0f 16 74 
        24 f8            movhpd -8(%r12), %xmm6                 #../src/feautrier.cpp:66.26
  01138 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:66.41
  0113c 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:66.41
  01140 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:66.57
  01143 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:66.57
  01147 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:67.57
  0114b 43 0f 11 6c d9 
        08               movups %xmm5, 8(%r9,%r11,8)            #../src/feautrier.cpp:66.7
  01151 43 0f 11 7c da 
        08               movups %xmm7, 8(%r10,%r11,8)           #../src/feautrier.cpp:67.7
  01157 49 83 c3 02      addq $2, %r11                          #../src/feautrier.cpp:64.5
  0115b 4c 3b df         cmpq %rdi, %r11                        #../src/feautrier.cpp:64.5
  0115e 72 b0            jb ..B6.46 # Prob 82%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r13 r14 r15 xmm0 xmm1 xmm2
..B6.47:                        # Preds ..B6.46
                                # Execution count [7.56e-01]
  01160 4c 8b 6c 24 08   movq 8(%rsp), %r13                     #[spill]
  01165 4c 8b 64 24 10   movq 16(%rsp), %r12                    #[spill]
                                # LOE rdx rcx rbx rbp rsi rdi r8 r12 r13 r14 r15 xmm1 xmm2
..B6.48:                        # Preds ..B6.47 ..B6.193
                                # Execution count [8.40e-01]
  0116a 49 89 fa         movq %rdi, %r10                        #../src/feautrier.cpp:64.5
  0116d 49 f7 da         negq %r10                              #../src/feautrier.cpp:64.5
  01170 48 3b f9         cmpq %rcx, %rdi                        #../src/feautrier.cpp:64.5
  01173 73 52            jae ..B6.52 # Prob 9%                  #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp rsi rdi r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.49:                        # Preds ..B6.48
                                # Execution count [7.56e-01]
  01175 4c 8d 1c f5 00 
        00 00 00         lea (,%rsi,8), %r11                    #../src/feautrier.cpp:66.26
  0117d 49 2b d3         subq %r11, %rdx                        #../src/feautrier.cpp:66.57
  01180 4d 8d 0c f7      lea (%r15,%rsi,8), %r9                 #../src/feautrier.cpp:66.7
  01184 49 8d 04 f6      lea (%r14,%rsi,8), %rax                #../src/feautrier.cpp:67.7
                                # LOE rax rdx rcx rbx rbp rdi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.50:                        # Preds ..B6.50 ..B6.49
                                # Execution count [4.20e+00]
  01188 f2 42 0f 10 5c 
        d2 f0            movsd -16(%rdx,%r10,8), %xmm3          #../src/feautrier.cpp:66.26
  0118f 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:66.41
  01192 f2 42 0f 10 6c 
        d2 e8            movsd -24(%rdx,%r10,8), %xmm5          #../src/feautrier.cpp:66.41
  01199 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:66.41
  0119c 49 ff ca         decq %r10                              #../src/feautrier.cpp:64.5
  0119f f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:66.41
  011a3 f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:66.41
  011a7 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:66.57
  011aa f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:66.57
  011ae f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:67.57
  011b2 f2 41 0f 11 64 
        f9 08            movsd %xmm4, 8(%r9,%rdi,8)             #../src/feautrier.cpp:66.7
  011b9 f2 0f 11 74 f8 
        08               movsd %xmm6, 8(%rax,%rdi,8)            #../src/feautrier.cpp:67.7
  011bf 48 ff c7         incq %rdi                              #../src/feautrier.cpp:64.5
  011c2 48 3b f9         cmpq %rcx, %rdi                        #../src/feautrier.cpp:64.5
  011c5 72 c1            jb ..B6.50 # Prob 82%                  #../src/feautrier.cpp:64.5
                                # LOE rax rdx rcx rbx rbp rdi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.52:                        # Preds ..B6.26 ..B6.50 ..B6.192 ..B6.48
                                # Execution count [4.20e-01]
  011c7 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:72.12
  011ca 7f 0e            jg ..B6.54 # Prob 84%                  #../src/feautrier.cpp:72.12
  011cc e9 2a 09 00 00   jmp ..B6.154 # Prob 100%               #../src/feautrier.cpp:72.12
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.53:                        # Preds ..B6.43
                                # Execution count [3.78e-01]
  011d1 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:72.12
  011d4 0f 8e e4 09 00 
        00               jle ..B6.191 # Prob 16%                #../src/feautrier.cpp:72.12
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.54:                        # Preds ..B6.53 ..B6.52 ..B6.195
                                # Execution count [8.40e-01]
  011da 49 83 fd 01      cmpq $1, %r13                          #../src/feautrier.cpp:74.30
  011de 0f 8e 2c 03 00 
        00               jle ..B6.80 # Prob 50%                 #../src/feautrier.cpp:74.30
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.55:                        # Preds ..B6.54
                                # Execution count [7.56e-01]
  011e4 4d 8d 55 ff      lea -1(%r13), %r10                     #../src/feautrier.cpp:74.5
  011e8 49 83 fa 08      cmpq $8, %r10                          #../src/feautrier.cpp:74.5
  011ec 0f 8c b8 09 00 
        00               jl ..B6.189 # Prob 10%                 #../src/feautrier.cpp:74.5
                                # LOE rbx rbp r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.56:                        # Preds ..B6.55
                                # Execution count [7.56e-01]
  011f2 48 8b 44 24 48   movq 72(%rsp), %rax                    #../src/feautrier.cpp:74.5[spill]
  011f7 48 83 e0 0f      andq $15, %rax                         #../src/feautrier.cpp:74.5
  011fb 74 11            je ..B6.59 # Prob 50%                  #../src/feautrier.cpp:74.5
                                # LOE rax rbx rbp r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.57:                        # Preds ..B6.56
                                # Execution count [7.56e-01]
  011fd 48 a9 07 00 00 
        00               testq $7, %rax                         #../src/feautrier.cpp:74.5
  01203 0f 85 a1 09 00 
        00               jne ..B6.189 # Prob 10%                #../src/feautrier.cpp:74.5
                                # LOE rbx rbp r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.58:                        # Preds ..B6.57
                                # Execution count [3.78e-01]
  01209 b8 01 00 00 00   movl $1, %eax                          #../src/feautrier.cpp:74.5
                                # LOE rax rbx rbp r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.59:                        # Preds ..B6.58 ..B6.56
                                # Execution count [7.56e-01]
  0120e 48 8d 50 08      lea 8(%rax), %rdx                      #../src/feautrier.cpp:74.5
  01212 4c 3b d2         cmpq %rdx, %r10                        #../src/feautrier.cpp:74.5
  01215 0f 8c 8f 09 00 
        00               jl ..B6.189 # Prob 10%                 #../src/feautrier.cpp:74.5
                                # LOE rax rbx rbp r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.60:                        # Preds ..B6.59
                                # Execution count [8.40e-01]
  0121b 4d 89 d3         movq %r10, %r11                        #../src/feautrier.cpp:74.5
  0121e 33 d2            xorl %edx, %edx                        #../src/feautrier.cpp:74.5
  01220 4c 2b d8         subq %rax, %r11                        #../src/feautrier.cpp:74.5
  01223 4b 8d 0c e7      lea (%r15,%r12,8), %rcx                #../src/feautrier.cpp:76.7
  01227 49 83 e3 07      andq $7, %r11                          #../src/feautrier.cpp:74.5
  0122b 4f 8d 0c e6      lea (%r14,%r12,8), %r9                 #../src/feautrier.cpp:77.7
  0122f 49 f7 db         negq %r11                              #../src/feautrier.cpp:74.5
  01232 4d 03 da         addq %r10, %r11                        #../src/feautrier.cpp:74.5
  01235 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:74.5
  01238 76 3c            jbe ..B6.64 # Prob 9%                  #../src/feautrier.cpp:74.5
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.61:                        # Preds ..B6.60
                                # Execution count [7.56e-01]
  0123a 48 8b 74 24 48   movq 72(%rsp), %rsi                    #[spill]
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.62:                        # Preds ..B6.61 ..B6.62
                                # Execution count [4.20e+00]
  0123f f2 0f 10 1c d6   movsd (%rsi,%rdx,8), %xmm3             #../src/feautrier.cpp:76.26
  01244 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:76.38
  01247 f2 0f 10 6c d6 
        08               movsd 8(%rsi,%rdx,8), %xmm5            #../src/feautrier.cpp:76.38
  0124d 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:76.38
  01250 f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:76.38
  01254 f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:76.38
  01258 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:76.53
  0125b f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:76.53
  0125f f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:77.53
  01263 f2 0f 11 24 d1   movsd %xmm4, (%rcx,%rdx,8)             #../src/feautrier.cpp:76.7
  01268 f2 41 0f 11 34 
        d1               movsd %xmm6, (%r9,%rdx,8)              #../src/feautrier.cpp:77.7
  0126e 48 ff c2         incq %rdx                              #../src/feautrier.cpp:74.5
  01271 48 3b d0         cmpq %rax, %rdx                        #../src/feautrier.cpp:74.5
  01274 72 c9            jb ..B6.62 # Prob 82%                  #../src/feautrier.cpp:74.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.64:                        # Preds ..B6.62 ..B6.60
                                # Execution count [0.00e+00]
  01276 0f 10 05 fc ff 
        ff ff            movups .L_2il0floatpacket.61(%rip), %xmm0 #../src/feautrier.cpp:76.19
  0127d 49 8d 14 04      lea (%r12,%rax), %rdx                  #../src/feautrier.cpp:77.14
  01281 49 8d 34 d6      lea (%r14,%rdx,8), %rsi                #../src/feautrier.cpp:77.7
  01285 48 f7 c6 0f 00 
        00 00            testq $15, %rsi                        #../src/feautrier.cpp:74.5
  0128c 0f 84 e3 00 00 
        00               je ..B6.68 # Prob 60%                  #../src/feautrier.cpp:74.5
                                # LOE rax rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.65:                        # Preds ..B6.64
                                # Execution count [7.56e-01]
  01292 48 8b 54 24 48   movq 72(%rsp), %rdx                    #[spill]
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.66:                        # Preds ..B6.65 ..B6.66
                                # Execution count [4.20e+00]
  01297 0f 10 24 c2      movups (%rdx,%rax,8), %xmm4            #../src/feautrier.cpp:76.26
  0129b 0f 10 74 c2 08   movups 8(%rdx,%rax,8), %xmm6           #../src/feautrier.cpp:76.26
  012a0 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:76.38
  012a3 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:76.38
  012a6 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:76.38
  012aa 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:76.38
  012ae 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:76.53
  012b1 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:76.38
  012b5 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:76.53
  012b9 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:77.53
  012bd 0f 11 2c c1      movups %xmm5, (%rcx,%rax,8)            #../src/feautrier.cpp:76.7
  012c1 41 0f 11 3c c1   movups %xmm7, (%r9,%rax,8)             #../src/feautrier.cpp:77.7
  012c6 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:76.38
  012c9 44 0f 10 4c c2 
        10               movups 16(%rdx,%rax,8), %xmm9          #../src/feautrier.cpp:76.26
  012cf 44 0f 10 5c c2 
        18               movups 24(%rdx,%rax,8), %xmm11         #../src/feautrier.cpp:76.26
  012d5 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:76.38
  012d9 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:76.38
  012de 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:76.38
  012e3 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:76.53
  012e7 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:76.53
  012ec 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:77.53
  012f1 44 0f 11 54 c1 
        10               movups %xmm10, 16(%rcx,%rax,8)         #../src/feautrier.cpp:76.7
  012f7 45 0f 11 64 c1 
        10               movups %xmm12, 16(%r9,%rax,8)          #../src/feautrier.cpp:77.7
  012fd 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:76.38
  01301 44 0f 10 74 c2 
        20               movups 32(%rdx,%rax,8), %xmm14         #../src/feautrier.cpp:76.26
  01307 0f 10 5c c2 28   movups 40(%rdx,%rax,8), %xmm3          #../src/feautrier.cpp:76.26
  0130c 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:76.38
  01310 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:76.38
  01315 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:76.38
  0131a 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:76.53
  0131e 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:76.53
  01323 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:77.53
  01327 44 0f 11 7c c1 
        20               movups %xmm15, 32(%rcx,%rax,8)         #../src/feautrier.cpp:76.7
  0132d 41 0f 11 64 c1 
        20               movups %xmm4, 32(%r9,%rax,8)           #../src/feautrier.cpp:77.7
  01333 0f 10 74 c2 30   movups 48(%rdx,%rax,8), %xmm6          #../src/feautrier.cpp:76.26
  01338 44 0f 10 44 c2 
        38               movups 56(%rdx,%rax,8), %xmm8          #../src/feautrier.cpp:76.26
  0133e 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:76.38
  01341 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:76.38
  01346 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:76.38
  0134b 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:76.53
  0134f 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:76.53
  01353 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:77.53
  01358 0f 11 7c c1 30   movups %xmm7, 48(%rcx,%rax,8)          #../src/feautrier.cpp:76.7
  0135d 45 0f 11 4c c1 
        30               movups %xmm9, 48(%r9,%rax,8)           #../src/feautrier.cpp:77.7
  01363 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:74.5
  01367 49 3b c3         cmpq %r11, %rax                        #../src/feautrier.cpp:74.5
  0136a 0f 82 27 ff ff 
        ff               jb ..B6.66 # Prob 82%                  #../src/feautrier.cpp:74.5
  01370 e9 de 00 00 00   jmp ..B6.71 # Prob 100%                #../src/feautrier.cpp:74.5
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.68:                        # Preds ..B6.64
                                # Execution count [7.56e-01]
  01375 48 8b 54 24 48   movq 72(%rsp), %rdx                    #[spill]
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.69:                        # Preds ..B6.68 ..B6.69
                                # Execution count [4.20e+00]
  0137a 0f 10 24 c2      movups (%rdx,%rax,8), %xmm4            #../src/feautrier.cpp:76.26
  0137e 0f 10 74 c2 08   movups 8(%rdx,%rax,8), %xmm6           #../src/feautrier.cpp:76.26
  01383 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:76.38
  01386 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:76.38
  01389 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:76.38
  0138d 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:76.38
  01391 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:76.53
  01394 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:76.38
  01398 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:76.53
  0139c 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:77.53
  013a0 0f 11 2c c1      movups %xmm5, (%rcx,%rax,8)            #../src/feautrier.cpp:76.7
  013a4 41 0f 11 3c c1   movups %xmm7, (%r9,%rax,8)             #../src/feautrier.cpp:77.7
  013a9 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:76.38
  013ac 44 0f 10 4c c2 
        10               movups 16(%rdx,%rax,8), %xmm9          #../src/feautrier.cpp:76.26
  013b2 44 0f 10 5c c2 
        18               movups 24(%rdx,%rax,8), %xmm11         #../src/feautrier.cpp:76.26
  013b8 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:76.38
  013bc 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:76.38
  013c1 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:76.38
  013c6 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:76.53
  013ca 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:76.53
  013cf 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:77.53
  013d4 44 0f 11 54 c1 
        10               movups %xmm10, 16(%rcx,%rax,8)         #../src/feautrier.cpp:76.7
  013da 45 0f 11 64 c1 
        10               movups %xmm12, 16(%r9,%rax,8)          #../src/feautrier.cpp:77.7
  013e0 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:76.38
  013e4 44 0f 10 74 c2 
        20               movups 32(%rdx,%rax,8), %xmm14         #../src/feautrier.cpp:76.26
  013ea 0f 10 5c c2 28   movups 40(%rdx,%rax,8), %xmm3          #../src/feautrier.cpp:76.26
  013ef 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:76.38
  013f3 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:76.38
  013f8 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:76.38
  013fd 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:76.53
  01401 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:76.53
  01406 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:77.53
  0140a 44 0f 11 7c c1 
        20               movups %xmm15, 32(%rcx,%rax,8)         #../src/feautrier.cpp:76.7
  01410 41 0f 11 64 c1 
        20               movups %xmm4, 32(%r9,%rax,8)           #../src/feautrier.cpp:77.7
  01416 0f 10 74 c2 30   movups 48(%rdx,%rax,8), %xmm6          #../src/feautrier.cpp:76.26
  0141b 44 0f 10 44 c2 
        38               movups 56(%rdx,%rax,8), %xmm8          #../src/feautrier.cpp:76.26
  01421 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:76.38
  01424 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:76.38
  01429 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:76.38
  0142e 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:76.53
  01432 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:76.53
  01436 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:77.53
  0143b 0f 11 7c c1 30   movups %xmm7, 48(%rcx,%rax,8)          #../src/feautrier.cpp:76.7
  01440 45 0f 11 4c c1 
        30               movups %xmm9, 48(%r9,%rax,8)           #../src/feautrier.cpp:77.7
  01446 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:74.5
  0144a 49 3b c3         cmpq %r11, %rax                        #../src/feautrier.cpp:74.5
  0144d 0f 82 27 ff ff 
        ff               jb ..B6.69 # Prob 82%                  #../src/feautrier.cpp:74.5
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.71:                        # Preds ..B6.69 ..B6.66
                                # Execution count [7.56e-01]
  01453 49 8d 43 01      lea 1(%r11), %rax                      #../src/feautrier.cpp:74.5
  01457 49 3b c2         cmpq %r10, %rax                        #../src/feautrier.cpp:74.5
  0145a 0f 87 b0 00 00 
        00               ja ..B6.80 # Prob 50%                  #../src/feautrier.cpp:74.5
                                # LOE rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.72:                        # Preds ..B6.71
                                # Execution count [7.56e-01]
  01460 4d 2b d3         subq %r11, %r10                        #../src/feautrier.cpp:74.5
  01463 49 83 fa 02      cmpq $2, %r10                          #../src/feautrier.cpp:74.5
  01467 0f 8c 4a 07 00 
        00               jl ..B6.190 # Prob 10%                 #../src/feautrier.cpp:74.5
                                # LOE rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.73:                        # Preds ..B6.72
                                # Execution count [7.56e-01]
  0146d 4c 89 d6         movq %r10, %rsi                        #../src/feautrier.cpp:74.5
  01470 33 d2            xorl %edx, %edx                        #../src/feautrier.cpp:74.5
  01472 48 8b 7c 24 48   movq 72(%rsp), %rdi                    #../src/feautrier.cpp:77.7[spill]
  01477 48 83 e6 fe      andq $-2, %rsi                         #../src/feautrier.cpp:74.5
  0147b 4a 8d 04 d9      lea (%rcx,%r11,8), %rax                #../src/feautrier.cpp:76.7
  0147f 4b 8d 0c d9      lea (%r9,%r11,8), %rcx                 #../src/feautrier.cpp:77.7
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.74:                        # Preds ..B6.74 ..B6.73
                                # Execution count [4.20e+00]
  01483 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:76.38
  01486 4e 8d 0c 1a      lea (%rdx,%r11), %r9                   #../src/feautrier.cpp:76.26
  0148a 42 0f 10 24 cf   movups (%rdi,%r9,8), %xmm4             #../src/feautrier.cpp:76.26
  0148f 42 0f 10 74 cf 
        08               movups 8(%rdi,%r9,8), %xmm6            #../src/feautrier.cpp:76.26
  01495 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:76.38
  01498 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:76.38
  0149c 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:76.38
  014a0 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:76.53
  014a3 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:76.53
  014a7 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:77.53
  014ab 0f 11 2c d0      movups %xmm5, (%rax,%rdx,8)            #../src/feautrier.cpp:76.7
  014af 0f 11 3c d1      movups %xmm7, (%rcx,%rdx,8)            #../src/feautrier.cpp:77.7
  014b3 48 83 c2 02      addq $2, %rdx                          #../src/feautrier.cpp:74.5
  014b7 48 3b d6         cmpq %rsi, %rdx                        #../src/feautrier.cpp:74.5
  014ba 72 c7            jb ..B6.74 # Prob 82%                  #../src/feautrier.cpp:74.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.76:                        # Preds ..B6.74 ..B6.190
                                # Execution count [8.40e-01]
  014bc 49 3b f2         cmpq %r10, %rsi                        #../src/feautrier.cpp:74.5
  014bf 73 4f            jae ..B6.80 # Prob 9%                  #../src/feautrier.cpp:74.5
                                # LOE rbx rbp rsi r8 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.77:                        # Preds ..B6.76
                                # Execution count [7.56e-01]
  014c1 48 8b 44 24 48   movq 72(%rsp), %rax                    #../src/feautrier.cpp:76.53[spill]
  014c6 4b 8d 0c e7      lea (%r15,%r12,8), %rcx                #../src/feautrier.cpp:76.7
  014ca 4b 8d 14 e6      lea (%r14,%r12,8), %rdx                #../src/feautrier.cpp:77.7
  014ce 4a 8d 0c d9      lea (%rcx,%r11,8), %rcx                #../src/feautrier.cpp:76.7
  014d2 4a 8d 14 da      lea (%rdx,%r11,8), %rdx                #../src/feautrier.cpp:77.7
  014d6 4a 8d 04 d8      lea (%rax,%r11,8), %rax                #../src/feautrier.cpp:76.53
                                # LOE rax rdx rcx rbx rbp rsi r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.78:                        # Preds ..B6.78 ..B6.77
                                # Execution count [4.20e+00]
  014da f2 0f 10 1c f0   movsd (%rax,%rsi,8), %xmm3             #../src/feautrier.cpp:76.26
  014df 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:76.38
  014e2 f2 0f 10 6c f0 
        08               movsd 8(%rax,%rsi,8), %xmm5            #../src/feautrier.cpp:76.38
  014e8 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:76.38
  014eb f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:76.38
  014ef f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:76.38
  014f3 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:76.53
  014f6 f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:76.53
  014fa f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:77.53
  014fe f2 0f 11 24 f1   movsd %xmm4, (%rcx,%rsi,8)             #../src/feautrier.cpp:76.7
  01503 f2 0f 11 34 f2   movsd %xmm6, (%rdx,%rsi,8)             #../src/feautrier.cpp:77.7
  01508 48 ff c6         incq %rsi                              #../src/feautrier.cpp:74.5
  0150b 49 3b f2         cmpq %r10, %rsi                        #../src/feautrier.cpp:74.5
  0150e 72 ca            jb ..B6.78 # Prob 82%                  #../src/feautrier.cpp:74.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B6.80:                        # Preds ..B6.78 ..B6.71 ..B6.54 ..B6.189 ..B6.76
                                #      
                                # Execution count [8.40e-01]
  01510 48 8b 54 24 48   movq 72(%rsp), %rdx                    #../src/feautrier.cpp:80.21[spill]
  01515 0f 28 e1         movaps %xmm1, %xmm4                    #../src/feautrier.cpp:80.21
  01518 49 8d 04 df      lea (%r15,%rbx,8), %rax                #../src/feautrier.cpp:80.5
  0151c 66 0f ef db      pxor %xmm3, %xmm3                      #../src/feautrier.cpp:81.5
  01520 48 8d 0c dd 00 
        00 00 00         lea (,%rbx,8), %rcx                    #../src/feautrier.cpp:80.5
  01528 f2 42 0f 10 44 
        ea f8            movsd -8(%rdx,%r13,8), %xmm0           #../src/feautrier.cpp:80.21
  0152f f2 0f 5e e0      divsd %xmm0, %xmm4                     #../src/feautrier.cpp:80.21
  01533 0f 28 ec         movaps %xmm4, %xmm5                    #../src/feautrier.cpp:80.35
  01536 f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:83.29
  0153a f2 0f 5e e8      divsd %xmm0, %xmm5                     #../src/feautrier.cpp:80.35
  0153e f2 0f 11 68 f8   movsd %xmm5, -8(%rax)                  #../src/feautrier.cpp:80.5
  01543 f2 0f 58 ec      addsd %xmm4, %xmm5                     #../src/feautrier.cpp:83.63
  01547 f2 41 0f 11 5c 
        0e f8            movsd %xmm3, -8(%r14,%rcx)             #../src/feautrier.cpp:81.5
  0154e f2 0f 11 64 24 
        40               movsd %xmm4, 64(%rsp)                  #../src/feautrier.cpp:83.29[spill]
  01554 f2 0f 11 6c 24 
        38               movsd %xmm5, 56(%rsp)                  #../src/feautrier.cpp:83.63[spill]
  0155a 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:88.14
  0155d 75 39            jne ..B6.82 # Prob 50%                 #../src/feautrier.cpp:88.14
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.81:                        # Preds ..B6.80 ..B6.197
                                # Execution count [5.00e-01]
  0155f 48 8b 54 24 48   movq 72(%rsp), %rdx                    #../src/feautrier.cpp:91.16[spill]
  01564 0f 28 e2         movaps %xmm2, %xmm4                    #../src/feautrier.cpp:93.27
  01567 66 0f ef c0      pxor %xmm0, %xmm0                      #../src/feautrier.cpp:90.5
  0156b f2 0f 10 1a      movsd (%rdx), %xmm3                    #../src/feautrier.cpp:91.16
  0156f f2 0f 5e cb      divsd %xmm3, %xmm1                     #../src/feautrier.cpp:91.16
  01573 0f 28 e9         movaps %xmm1, %xmm5                    #../src/feautrier.cpp:91.26
  01576 f2 0f 58 e1      addsd %xmm1, %xmm4                     #../src/feautrier.cpp:93.27
  0157a f2 0f 5e eb      divsd %xmm3, %xmm5                     #../src/feautrier.cpp:91.26
  0157e f2 41 0f 11 2e   movsd %xmm5, (%r14)                    #../src/feautrier.cpp:91.5
  01583 f2 0f 58 ec      addsd %xmm4, %xmm5                     #../src/feautrier.cpp:93.53
  01587 f2 41 0f 11 07   movsd %xmm0, (%r15)                    #../src/feautrier.cpp:90.5
  0158c f2 0f 11 64 24 
        30               movsd %xmm4, 48(%rsp)                  #../src/feautrier.cpp:93.27[spill]
  01592 f2 0f 11 6c 24 
        28               movsd %xmm5, 40(%rsp)                  #../src/feautrier.cpp:93.53[spill]
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.82:                        # Preds ..B6.80 ..B6.81 ..B6.198
                                # Execution count [8.72e-01]
  01598 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:98.13
  0159b 75 50            jne ..B6.85 # Prob 50%                 #../src/feautrier.cpp:98.13
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.83:                        # Preds ..B6.82
                                # Execution count [4.36e-01]
  0159d f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:100.17
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.84:                        # Preds ..B6.154 ..B6.191 ..B6.83
                                # Execution count [5.00e-01]
  015a5 48 8b 54 24 58   movq 88(%rsp), %rdx                    #../src/feautrier.cpp:100.21[spill]
  015aa 0f 28 ea         movaps %xmm2, %xmm5                    #../src/feautrier.cpp:103.29
  015ad 48 83 7c 24 18 
        00               cmpq $0, 24(%rsp)                      #../src/feautrier.cpp:110.30[spill]
  015b3 66 0f ef e4      pxor %xmm4, %xmm4                      #../src/feautrier.cpp:101.5
  015b7 f2 0f 10 1a      movsd (%rdx), %xmm3                    #../src/feautrier.cpp:100.21
  015bb f2 0f 5e cb      divsd %xmm3, %xmm1                     #../src/feautrier.cpp:100.21
  015bf 0f 28 c1         movaps %xmm1, %xmm0                    #../src/feautrier.cpp:100.32
  015c2 f2 0f 58 e9      addsd %xmm1, %xmm5                     #../src/feautrier.cpp:103.29
  015c6 f2 0f 5e c3      divsd %xmm3, %xmm0                     #../src/feautrier.cpp:100.32
  015ca f2 0f 11 6c 24 
        40               movsd %xmm5, 64(%rsp)                  #../src/feautrier.cpp:103.29[spill]
  015d0 f2 0f 58 e8      addsd %xmm0, %xmm5                     #../src/feautrier.cpp:103.57
  015d4 f2 0f 11 40 f8   movsd %xmm0, -8(%rax)                  #../src/feautrier.cpp:100.5
  015d9 f2 42 0f 11 64 
        31 f8            movsd %xmm4, -8(%rcx,%r14)             #../src/feautrier.cpp:101.5
  015e0 f2 0f 11 6c 24 
        38               movsd %xmm5, 56(%rsp)                  #../src/feautrier.cpp:103.57[spill]
  015e6 7d 0d            jge ..B6.86 # Prob 50%                 #../src/feautrier.cpp:110.30
  015e8 e9 f8 00 00 00   jmp ..B6.98 # Prob 100%                #../src/feautrier.cpp:110.30
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2
..B6.85:                        # Preds ..B6.82
                                # Execution count [4.36e-01]
  015ed 48 83 7c 24 18 
        00               cmpq $0, 24(%rsp)                      #../src/feautrier.cpp:110.30[spill]
  015f3 7c 6f            jl ..B6.92 # Prob 50%                  #../src/feautrier.cpp:110.30
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.86:                        # Preds ..B6.155 ..B6.191 ..B6.84 ..B6.85
                                # Execution count [1.00e+00]
  015f5 4d 89 e1         movq %r12, %r9                         #../src/feautrier.cpp:110.3
  015f8 33 f6            xorl %esi, %esi                        #../src/feautrier.cpp:110.3
  015fa 49 c1 e9 3f      shrq $63, %r9                          #../src/feautrier.cpp:110.3
  015fe 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:112.5
  01601 4d 03 cc         addq %r12, %r9                         #../src/feautrier.cpp:25.5
  01604 ba 01 00 00 00   movl $1, %edx                          #../src/feautrier.cpp:110.3
  01609 49 d1 f9         sarq $1, %r9                           #../src/feautrier.cpp:25.5
  0160c 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:112.5
  0160f 4d 85 c9         testq %r9, %r9                         #../src/feautrier.cpp:110.3
  01612 76 34            jbe ..B6.90 # Prob 2%                  #../src/feautrier.cpp:110.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.87:                        # Preds ..B6.86
                                # Execution count [9.79e-01]
  01614 48 8b 54 24 50   movq 80(%rsp), %rdx                    #../src/feautrier.cpp:112.19[spill]
  01619 4a 8d 14 e2      lea (%rdx,%r12,8), %rdx                #../src/feautrier.cpp:112.19
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.88:                        # Preds ..B6.88 ..B6.87
                                # Execution count [2.72e+00]
  0161d 49 8b 7c 13 f8   movq -8(%r11,%rdx), %rdi               #../src/feautrier.cpp:112.19
  01622 48 ff c6         incq %rsi                              #../src/feautrier.cpp:110.3
  01625 4b 89 3c 02      movq %rdi, (%r10,%r8)                  #../src/feautrier.cpp:112.5
  01629 49 8b 7c 13 f0   movq -16(%r11,%rdx), %rdi              #../src/feautrier.cpp:112.19
  0162e 49 83 c3 f0      addq $-16, %r11                        #../src/feautrier.cpp:110.3
  01632 4b 89 7c 02 08   movq %rdi, 8(%r10,%r8)                 #../src/feautrier.cpp:112.5
  01637 49 83 c2 10      addq $16, %r10                         #../src/feautrier.cpp:110.3
  0163b 49 3b f1         cmpq %r9, %rsi                         #../src/feautrier.cpp:110.3
  0163e 72 dd            jb ..B6.88 # Prob 63%                  #../src/feautrier.cpp:110.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.89:                        # Preds ..B6.88
                                # Execution count [9.79e-01]
  01640 48 8d 14 75 01 
        00 00 00         lea 1(,%rsi,2), %rdx                   #../src/feautrier.cpp:112.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.90:                        # Preds ..B6.89 ..B6.86
                                # Execution count [1.00e+00]
  01648 48 ff ca         decq %rdx                              #../src/feautrier.cpp:112.5
  0164b 49 3b d4         cmpq %r12, %rdx                        #../src/feautrier.cpp:110.3
  0164e 73 14            jae ..B6.92 # Prob 2%                  #../src/feautrier.cpp:110.3
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.91:                        # Preds ..B6.90
                                # Execution count [9.79e-01]
  01650 4d 89 e1         movq %r12, %r9                         #../src/feautrier.cpp:112.5
  01653 4c 2b ca         subq %rdx, %r9                         #../src/feautrier.cpp:112.5
  01656 48 8b 74 24 50   movq 80(%rsp), %rsi                    #../src/feautrier.cpp:112.19[spill]
  0165b 4e 8b 54 ce f8   movq -8(%rsi,%r9,8), %r10              #../src/feautrier.cpp:112.19
  01660 4d 89 14 d0      movq %r10, (%r8,%rdx,8)                #../src/feautrier.cpp:112.5
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.92:                        # Preds ..B6.91 ..B6.90 ..B6.85
                                # Execution count [7.18e-01]
  01664 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:115.24
  01667 7e 77            jle ..B6.97 # Prob 50%                 #../src/feautrier.cpp:115.24
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.93:                        # Preds ..B6.92
                                # Execution count [5.00e-03]
  01669 49 83 fd 0c      cmpq $12, %r13                         #../src/feautrier.cpp:115.3
  0166d 0f 8e ab 04 00 
        00               jle ..B6.158 # Prob 10%                #../src/feautrier.cpp:115.3
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.94:                        # Preds ..B6.93
                                # Execution count [1.00e+00]
  01673 41 b9 01 00 00 
        00               movl $1, %r9d                          #../src/feautrier.cpp:115.3
  01679 4b 8d 3c e0      lea (%r8,%r12,8), %rdi                 #../src/feautrier.cpp:117.5
  0167d 48 89 fe         movq %rdi, %rsi                        #../src/feautrier.cpp:115.3
  01680 4a 8d 14 ed 00 
        00 00 00         lea (,%r13,8), %rdx                    #../src/feautrier.cpp:115.3
  01688 48 2b 74 24 20   subq 32(%rsp), %rsi                    #../src/feautrier.cpp:115.3[spill]
  0168d 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:115.3
  01690 48 3b f2         cmpq %rdx, %rsi                        #../src/feautrier.cpp:115.3
  01693 45 0f 4f d9      cmovg %r9d, %r11d                      #../src/feautrier.cpp:115.3
  01697 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:115.3
  0169a 48 f7 de         negq %rsi                              #../src/feautrier.cpp:115.3
  0169d 48 3b f2         cmpq %rdx, %rsi                        #../src/feautrier.cpp:115.3
  016a0 45 0f 4f d1      cmovg %r9d, %r10d                      #../src/feautrier.cpp:115.3
  016a4 45 0b da         orl %r10d, %r11d                       #../src/feautrier.cpp:115.3
  016a7 0f 84 71 04 00 
        00               je ..B6.158 # Prob 10%                 #../src/feautrier.cpp:115.3
                                # LOE rax rdx rcx rbx rbp rdi r8 r12 r13 r14 r15 xmm2
..B6.95:                        # Preds ..B6.94
                                # Execution count [1.00e+00]
  016ad 48 8b 74 24 20   movq 32(%rsp), %rsi                    #../src/feautrier.cpp:117.5[spill]
  016b2 48 89 04 24      movq %rax, (%rsp)                      #../src/feautrier.cpp:117.5[spill]
  016b6 48 89 4c 24 08   movq %rcx, 8(%rsp)                     #../src/feautrier.cpp:117.5[spill]
  016bb e8 fc ff ff ff   call _intel_fast_memcpy                #../src/feautrier.cpp:117.5
                                # LOE rbx rbp r12 r13 r14 r15
..B6.96:                        # Preds ..B6.95
                                # Execution count [1.00e+00]
  016c0 48 8b 04 24      movq (%rsp), %rax                      #[spill]
  016c4 48 8b 4c 24 08   movq 8(%rsp), %rcx                     #[spill]
  016c9 4c 8b 84 24 a0 
        00 00 00         movq 160(%rsp), %r8                    #
  016d1 f2 0f 10 15 fc 
        ff ff ff         movsd .L_2il0floatpacket.62(%rip), %xmm2 #
  016d9 f2 0f 10 40 f8   movsd -8(%rax), %xmm0                  #../src/feautrier.cpp:142.28
  016de eb 05            jmp ..B6.98 # Prob 100%                #../src/feautrier.cpp:142.28
                                # LOE rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2
..B6.97:                        # Preds ..B6.162 ..B6.155 ..B6.92
                                # Execution count [3.59e-01]
  016e0 f2 0f 10 40 f8   movsd -8(%rax), %xmm0                  #../src/feautrier.cpp:142.28
                                # LOE rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2
..B6.98:                        # Preds ..B6.96 ..B6.163 ..B6.84 ..B6.97
                                # Execution count [1.00e+00]
  016e5 f2 41 0f 10 08   movsd (%r8), %xmm1                     #../src/feautrier.cpp:129.10
  016ea 33 d2            xorl %edx, %edx                        #../src/feautrier.cpp:134.3
  016ec f2 0f 10 64 24 
        30               movsd 48(%rsp), %xmm4                  #../src/feautrier.cpp:131.22[spill]
  016f2 f2 0f 5e 4c 24 
        28               divsd 40(%rsp), %xmm1                  #../src/feautrier.cpp:129.17[spill]
  016f8 f2 41 0f 5e 26   divsd (%r14), %xmm4                    #../src/feautrier.cpp:131.22
  016fd 48 8b 44 24 60   movq 96(%rsp), %rax                    #../src/feautrier.cpp:131.3[spill]
  01702 f2 41 0f 11 08   movsd %xmm1, (%r8)                     #../src/feautrier.cpp:129.3
  01707 f2 0f 11 20      movsd %xmm4, (%rax)                    #../src/feautrier.cpp:131.3
  0170b 4b 8d 44 25 fe   lea -2(%r13,%r12), %rax                #../src/feautrier.cpp:134.29
  01710 48 83 fb 02      cmpq $2, %rbx                          #../src/feautrier.cpp:134.29
  01714 7e 6c            jle ..B6.102 # Prob 9%                 #../src/feautrier.cpp:134.29
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm1 xmm2 xmm4
..B6.99:                        # Preds ..B6.98
                                # Execution count [9.00e-01]
  01716 48 8b 74 24 60   movq 96(%rsp), %rsi                    #[spill]
  0171b 0f 1f 44 00 00   .align    16,0x90
                                # LOE rax rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.100:                       # Preds ..B6.100 ..B6.99
                                # Execution count [5.00e+00]
  01720 f2 41 0f 10 6c 
        d7 08            movsd 8(%r15,%rdx,8), %xmm5            #../src/feautrier.cpp:136.19
  01727 f2 0f 10 1c d6   movsd (%rsi,%rdx,8), %xmm3             #../src/feautrier.cpp:136.24
  0172c 0f 28 e5         movaps %xmm5, %xmm4                    #../src/feautrier.cpp:136.24
  0172f f2 0f 59 e3      mulsd %xmm3, %xmm4                     #../src/feautrier.cpp:136.24
  01733 f2 0f 58 da      addsd %xmm2, %xmm3                     #../src/feautrier.cpp:136.38
  01737 f2 0f 59 e9      mulsd %xmm1, %xmm5                     #../src/feautrier.cpp:138.25
  0173b f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:136.38
  0173f f2 41 0f 10 74 
        d6 08            movsd 8(%r14,%rdx,8), %xmm6            #../src/feautrier.cpp:136.49
  01746 f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:136.38
  0174a f2 41 0f 58 6c 
        d0 08            addsd 8(%r8,%rdx,8), %xmm5             #../src/feautrier.cpp:138.25
  01751 f2 0f 5e e6      divsd %xmm6, %xmm4                     #../src/feautrier.cpp:136.49
  01755 f2 0f 11 64 d6 
        08               movsd %xmm4, 8(%rsi,%rdx,8)            #../src/feautrier.cpp:136.5
  0175b f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:138.42
  0175f f2 0f 5e ec      divsd %xmm4, %xmm5                     #../src/feautrier.cpp:138.42
  01763 0f 28 cd         movaps %xmm5, %xmm1                    #../src/feautrier.cpp:138.50
  01766 f2 0f 5e ce      divsd %xmm6, %xmm1                     #../src/feautrier.cpp:138.50
  0176a f2 41 0f 11 4c 
        d0 08            movsd %xmm1, 8(%r8,%rdx,8)             #../src/feautrier.cpp:138.5
  01771 48 ff c2         incq %rdx                              #../src/feautrier.cpp:134.3
  01774 48 3b d0         cmpq %rax, %rdx                        #../src/feautrier.cpp:134.3
  01777 72 a7            jb ..B6.100 # Prob 82%                 #../src/feautrier.cpp:134.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.101:                       # Preds ..B6.100
                                # Execution count [9.00e-01]
  01779 48 8b 54 24 60   movq 96(%rsp), %rdx                    #../src/feautrier.cpp:157.27[spill]
  0177e f2 0f 10 22      movsd (%rdx), %xmm4                    #../src/feautrier.cpp:157.27
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2 xmm4
..B6.102:                       # Preds ..B6.98 ..B6.101
                                # Execution count [1.00e+00]
  01782 48 8b 54 24 60   movq 96(%rsp), %rdx                    #../src/feautrier.cpp:143.36[spill]
  01787 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:150.3
  0178a f2 0f 10 6c 24 
        38               movsd 56(%rsp), %xmm5                  #../src/feautrier.cpp:143.36[spill]
  01790 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:152.5
  01793 f2 41 0f 10 5c 
        08 f0            movsd -16(%r8,%rcx), %xmm3             #../src/feautrier.cpp:142.38
  0179a f2 0f 10 4c 0a 
        f0               movsd -16(%rdx,%rcx), %xmm1            #../src/feautrier.cpp:143.36
  017a0 f2 0f 59 e9      mulsd %xmm1, %xmm5                     #../src/feautrier.cpp:143.36
  017a4 f2 0f 58 ca      addsd %xmm2, %xmm1                     #../src/feautrier.cpp:143.56
  017a8 f2 0f 59 d8      mulsd %xmm0, %xmm3                     #../src/feautrier.cpp:142.38
  017ac f2 0f 10 74 24 
        40               movsd 64(%rsp), %xmm6                  #../src/feautrier.cpp:143.36[spill]
  017b2 f2 0f 58 ee      addsd %xmm6, %xmm5                     #../src/feautrier.cpp:143.36
  017b6 f2 41 0f 58 5c 
        08 f8            addsd -8(%r8,%rcx), %xmm3              #../src/feautrier.cpp:142.38
  017bd f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:145.29
  017c1 f2 0f 5e dd      divsd %xmm5, %xmm3                     #../src/feautrier.cpp:143.36
  017c5 f2 0f 59 d9      mulsd %xmm1, %xmm3                     #../src/feautrier.cpp:143.56
  017c9 f2 0f 11 6c 24 
        38               movsd %xmm5, 56(%rsp)                  #../src/feautrier.cpp:143.36[spill]
  017cf f2 41 0f 11 5c 
        08 f8            movsd %xmm3, -8(%r8,%rcx)              #../src/feautrier.cpp:142.3
  017d6 f2 0f 11 74 24 
        40               movsd %xmm6, 64(%rsp)                  #../src/feautrier.cpp:145.29[spill]
  017dc f2 0f 11 74 0d 
        f8               movsd %xmm6, -8(%rbp,%rcx)             #../src/feautrier.cpp:145.3
  017e2 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:150.29
  017e5 7e 79            jle ..B6.106 # Prob 2%                 #../src/feautrier.cpp:150.29
                                # LOE rax rdx rbx rbp r8 r10 r11 r12 r13 r14 r15 xmm1 xmm2 xmm3 xmm4 xmm6
..B6.103:                       # Preds ..B6.102
                                # Execution count [9.79e-01]
  017e7 48 89 d7         movq %rdx, %rdi                        #../src/feautrier.cpp:152.31
  017ea 4b 8d 54 25 00   lea (%r13,%r12), %rdx                  #../src/feautrier.cpp:152.31
  017ef 4c 8d 4c d5 00   lea (%rbp,%rdx,8), %r9                 #../src/feautrier.cpp:154.5
  017f4 49 8d 34 d7      lea (%r15,%rdx,8), %rsi                #../src/feautrier.cpp:154.47
  017f8 49 8d 0c d6      lea (%r14,%rdx,8), %rcx                #../src/feautrier.cpp:154.19
  017fc 48 8d 3c d7      lea (%rdi,%rdx,8), %rdi                #../src/feautrier.cpp:152.31
  01800 49 8d 14 d0      lea (%r8,%rdx,8), %rdx                 #../src/feautrier.cpp:152.12
  01804 0f 1f 44 00 00 
        0f 1f 80 00 00 
        00 00            .align    16,0x90
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2 xmm3 xmm4 xmm6
..B6.104:                       # Preds ..B6.104 ..B6.103
                                # Execution count [5.44e+00]
  01810 f2 42 0f 10 6c 
        d1 f0            movsd -16(%rcx,%r10,8), %xmm5          #../src/feautrier.cpp:154.19
  01817 49 ff c3         incq %r11                              #../src/feautrier.cpp:150.3
  0181a f2 0f 59 ee      mulsd %xmm6, %xmm5                     #../src/feautrier.cpp:154.24
  0181e f2 0f 58 f2      addsd %xmm2, %xmm6                     #../src/feautrier.cpp:154.36
  01822 f2 0f 5e ee      divsd %xmm6, %xmm5                     #../src/feautrier.cpp:154.36
  01826 f2 42 0f 10 44 
        d7 f0            movsd -16(%rdi,%r10,8), %xmm0          #../src/feautrier.cpp:152.31
  0182d f2 0f 58 ea      addsd %xmm2, %xmm5                     #../src/feautrier.cpp:154.36
  01831 f2 0f 58 c2      addsd %xmm2, %xmm0                     #../src/feautrier.cpp:152.31
  01835 f2 0f 5e d8      divsd %xmm0, %xmm3                     #../src/feautrier.cpp:152.31
  01839 0f 28 f5         movaps %xmm5, %xmm6                    #../src/feautrier.cpp:154.47
  0183c f2 42 0f 58 5c 
        d2 f0            addsd -16(%rdx,%r10,8), %xmm3          #../src/feautrier.cpp:152.31
  01843 f2 42 0f 5e 74 
        d6 f0            divsd -16(%rsi,%r10,8), %xmm6          #../src/feautrier.cpp:154.47
  0184a f2 42 0f 11 5c 
        d2 f0            movsd %xmm3, -16(%rdx,%r10,8)          #../src/feautrier.cpp:152.5
  01851 f2 43 0f 11 74 
        d1 f0            movsd %xmm6, -16(%r9,%r10,8)           #../src/feautrier.cpp:154.5
  01858 49 ff ca         decq %r10                              #../src/feautrier.cpp:150.3
  0185b 4c 3b d8         cmpq %rax, %r11                        #../src/feautrier.cpp:150.3
  0185e 72 b0            jb ..B6.104 # Prob 82%                 #../src/feautrier.cpp:150.3
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2 xmm3 xmm4 xmm6
..B6.106:                       # Preds ..B6.104 ..B6.102
                                # Execution count [9.90e-01]
  01860 f2 41 0f 10 40 
        08               movsd 8(%r8), %xmm0                    #../src/feautrier.cpp:157.17
  01866 f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:157.27
  0186a f2 0f 5e c4      divsd %xmm4, %xmm0                     #../src/feautrier.cpp:157.27
  0186e 4c 8b 94 24 a8 
        00 00 00         movq 168(%rsp), %r10                   #../src/feautrier.cpp:28.1
  01876 f2 41 0f 58 00   addsd (%r8), %xmm0                     #../src/feautrier.cpp:157.27
  0187b f2 41 0f 11 00   movsd %xmm0, (%r8)                     #../src/feautrier.cpp:157.3
  01880 4d 8b 42 08      movq 8(%r10), %r8                      #../src/feautrier.cpp:168.9
  01884 f2 0f 10 45 08   movsd 8(%rbp), %xmm0                   #../src/feautrier.cpp:168.24
  01889 4d 85 c0         testq %r8, %r8                         #../src/feautrier.cpp:168.9
  0188c 0f 8e ff 02 00 
        00               jle ..B6.166 # Prob 5%                 #../src/feautrier.cpp:168.9
                                # LOE rbx rbp r8 r10 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.107:                       # Preds ..B6.106
                                # Execution count [9.31e-01]
  01892 49 8b 4a 10      movq 16(%r10), %rcx                    #../src/feautrier.cpp:168.9
  01896 48 85 c9         testq %rcx, %rcx                       #../src/feautrier.cpp:168.9
  01899 0f 8e f2 02 00 
        00               jle ..B6.166 # Prob 0%                 #../src/feautrier.cpp:168.9
                                # LOE rcx rbx rbp r8 r10 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B6.108:                       # Preds ..B6.107
                                # Execution count [9.27e-01]
  0189f 0f 28 da         movaps %xmm2, %xmm3                    #../src/feautrier.cpp:168.24
  018a2 4b 8d 54 25 ff   lea -1(%r13,%r12), %rdx                #../src/feautrier.cpp:170.29
  018a7 4d 8b 0a         movq (%r10), %r9                       #../src/feautrier.cpp:168.9
  018aa b8 01 00 00 00   movl $1, %eax                          #../src/feautrier.cpp:170.15
  018af f2 0f 58 d8      addsd %xmm0, %xmm3                     #../src/feautrier.cpp:168.24
  018b3 f2 0f 59 44 24 
        28               mulsd 40(%rsp), %xmm0                  #../src/feautrier.cpp:168.48[spill]
  018b9 f2 0f 58 44 24 
        30               addsd 48(%rsp), %xmm0                  #../src/feautrier.cpp:168.48[spill]
  018bf f2 0f 5e d8      divsd %xmm0, %xmm3                     #../src/feautrier.cpp:168.48
  018c3 f2 41 0f 11 19   movsd %xmm3, (%r9)                     #../src/feautrier.cpp:168.9
  018c8 48 83 fb 02      cmpq $2, %rbx                          #../src/feautrier.cpp:170.29
  018cc 7e 70            jle ..B6.116 # Prob 10%                #../src/feautrier.cpp:170.29
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r14 r15 xmm1 xmm2
..B6.109:                       # Preds ..B6.108
                                # Execution count [1.20e+00]
  018ce 4c 8b 5c 24 60   movq 96(%rsp), %r11                    #[spill]
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r14 r15 xmm1 xmm2
..B6.110:                       # Preds ..B6.109 ..B6.114
                                # Execution count [5.00e+00]
  018d3 f2 0f 10 5c c5 
        08               movsd 8(%rbp,%rax,8), %xmm3            #../src/feautrier.cpp:172.26
  018d9 0f 28 ea         movaps %xmm2, %xmm5                    #../src/feautrier.cpp:172.26
  018dc f2 41 0f 10 04 
        c3               movsd (%r11,%rax,8), %xmm0             #../src/feautrier.cpp:172.37
  018e2 0f 28 e3         movaps %xmm3, %xmm4                    #../src/feautrier.cpp:172.44
  018e5 f2 0f 58 eb      addsd %xmm3, %xmm5                     #../src/feautrier.cpp:172.26
  018e9 f2 0f 59 d8      mulsd %xmm0, %xmm3                     #../src/feautrier.cpp:172.58
  018ed f2 0f 58 e0      addsd %xmm0, %xmm4                     #../src/feautrier.cpp:172.44
  018f1 f2 0f 58 e3      addsd %xmm3, %xmm4                     #../src/feautrier.cpp:172.58
  018f5 f2 0f 5e ec      divsd %xmm4, %xmm5                     #../src/feautrier.cpp:172.58
  018f9 f2 41 0f 5e 2c 
        c6               divsd (%r14,%rax,8), %xmm5             #../src/feautrier.cpp:172.68
  018ff 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:172.11
  01902 0f 8c 89 02 00 
        00               jl ..B6.166 # Prob 0%                  #../src/feautrier.cpp:172.11
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r14 r15 xmm1 xmm2 xmm5
..B6.111:                       # Preds ..B6.110
                                # Execution count [4.95e+00]
  01908 49 3b 42 08      cmpq 8(%r10), %rax                     #../src/feautrier.cpp:172.11
  0190c 0f 8d 7f 02 00 
        00               jge ..B6.166 # Prob 5%                 #../src/feautrier.cpp:172.11
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r14 r15 xmm1 xmm2 xmm5
..B6.112:                       # Preds ..B6.111
                                # Execution count [4.70e+00]
  01912 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:172.11
  01915 0f 8c 76 02 00 
        00               jl ..B6.166 # Prob 0%                  #../src/feautrier.cpp:172.11
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r14 r15 xmm1 xmm2 xmm5
..B6.113:                       # Preds ..B6.112
                                # Execution count [4.66e+00]
  0191b 49 3b 42 10      cmpq 16(%r10), %rax                    #../src/feautrier.cpp:172.11
  0191f 0f 8d 6c 02 00 
        00               jge ..B6.166 # Prob 0%                 #../src/feautrier.cpp:172.11
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r14 r15 xmm1 xmm2 xmm5
..B6.114:                       # Preds ..B6.113
                                # Execution count [4.63e+00]
  01925 49 8b 72 18      movq 24(%r10), %rsi                    #../src/feautrier.cpp:172.11
  01929 48 0f af f0      imulq %rax, %rsi                       #../src/feautrier.cpp:172.11
  0192d 48 03 f0         addq %rax, %rsi                        #../src/feautrier.cpp:172.68
  01930 48 ff c0         incq %rax                              #../src/feautrier.cpp:170.32
  01933 f2 41 0f 11 2c 
        f1               movsd %xmm5, (%r9,%rsi,8)              #../src/feautrier.cpp:172.11
  01939 48 3b c2         cmpq %rdx, %rax                        #../src/feautrier.cpp:170.29
  0193c 7c 95            jl ..B6.110 # Prob 82%                 #../src/feautrier.cpp:170.29
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r14 r15 xmm1 xmm2
..B6.116:                       # Preds ..B6.114 ..B6.108
                                # Execution count [1.00e+00]
  0193e 48 85 d2         testq %rdx, %rdx                       #../src/feautrier.cpp:175.9
  01941 0f 8c 4a 02 00 
        00               jl ..B6.166 # Prob 0%                  #../src/feautrier.cpp:175.9
                                # LOE rdx rcx rbx rbp r8 r9 r10 r14 r15 xmm1 xmm2
..B6.117:                       # Preds ..B6.116
                                # Execution count [9.90e-01]
  01947 49 3b d0         cmpq %r8, %rdx                         #../src/feautrier.cpp:175.9
  0194a 0f 8d 41 02 00 
        00               jge ..B6.166 # Prob 5%                 #../src/feautrier.cpp:175.9
                                # LOE rdx rcx rbx rbp r8 r9 r10 r14 r15 xmm1 xmm2
..B6.118:                       # Preds ..B6.117
                                # Execution count [9.41e-01]
  01950 48 85 d2         testq %rdx, %rdx                       #../src/feautrier.cpp:175.9
  01953 0f 8c 38 02 00 
        00               jl ..B6.166 # Prob 0%                  #../src/feautrier.cpp:175.9
                                # LOE rdx rcx rbx rbp r8 r9 r10 r14 r15 xmm1 xmm2
..B6.119:                       # Preds ..B6.118
                                # Execution count [9.31e-01]
  01959 48 3b d1         cmpq %rcx, %rdx                        #../src/feautrier.cpp:175.9
  0195c 0f 8d 2f 02 00 
        00               jge ..B6.166 # Prob 0%                 #../src/feautrier.cpp:175.9
                                # LOE rdx rbx rbp r8 r9 r10 r14 r15 xmm1 xmm2
..B6.120:                       # Preds ..B6.119
                                # Execution count [9.27e-01]
  01962 f2 0f 5e 4c 24 
        38               divsd 56(%rsp), %xmm1                  #../src/feautrier.cpp:175.66[spill]
  01968 49 8b 72 18      movq 24(%r10), %rsi                    #../src/feautrier.cpp:175.9
  0196c 48 c7 c0 ff ff 
        ff ff            movq $-1, %rax                         #../src/feautrier.cpp:180.15
  01973 48 0f af d6      imulq %rsi, %rdx                       #../src/feautrier.cpp:175.9
  01977 48 03 d3         addq %rbx, %rdx                        #../src/feautrier.cpp:175.9
  0197a 48 8b 8c 24 b0 
        00 00 00         movq 176(%rsp), %rcx                   #../src/feautrier.cpp:28.1
  01982 f2 41 0f 11 4c 
        d1 f8            movsd %xmm1, -8(%r9,%rdx,8)            #../src/feautrier.cpp:175.9
  01989 ba 01 00 00 00   movl $1, %edx                          #../src/feautrier.cpp:180.15
  0198e 48 83 f9 01      cmpq $1, %rcx                          #../src/feautrier.cpp:180.24
  01992 0f 8e 21 01 00 
        00               jle ..B6.146 # Prob 10%                #../src/feautrier.cpp:180.24
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r14 r15 xmm2
..B6.121:                       # Preds ..B6.120
                                # Execution count [9.00e-01]
  01998 4c 89 3c 24      movq %r15, (%rsp)                      #[spill]
  0199c 4c 8b 6c 24 60   movq 96(%rsp), %r13                    #[spill]
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r13 r14 xmm2
..B6.122:                       # Preds ..B6.132 ..B6.121
                                # Execution count [5.00e+00]
  019a1 45 33 e4         xorl %r12d, %r12d                      #../src/feautrier.cpp:182.17
  019a4 4c 8d 1c 03      lea (%rbx,%rax), %r11                  #../src/feautrier.cpp:30.20
  019a8 4d 85 db         testq %r11, %r11                       #../src/feautrier.cpp:182.31
  019ab 7e 68            jle ..B6.132 # Prob 10%                #../src/feautrier.cpp:182.31
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 xmm2
..B6.124:                       # Preds ..B6.122 ..B6.130
                                # Execution count [2.50e+01]
  019ad 4c 89 e7         movq %r12, %rdi                        #../src/feautrier.cpp:184.32
  019b0 4e 8d 3c 22      lea (%rdx,%r12), %r15                  #../src/feautrier.cpp:184.36
  019b4 48 ff c7         incq %rdi                              #../src/feautrier.cpp:184.32
  019b7 0f 88 d4 01 00 
        00               js ..B6.166 # Prob 0%                  #../src/feautrier.cpp:184.29
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.125:                       # Preds ..B6.124
                                # Execution count [2.48e+01]
  019bd 49 3b f8         cmpq %r8, %rdi                         #../src/feautrier.cpp:184.29
  019c0 0f 8d cb 01 00 
        00               jge ..B6.166 # Prob 5%                 #../src/feautrier.cpp:184.29
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.126:                       # Preds ..B6.125
                                # Execution count [2.35e+01]
  019c6 4d 85 ff         testq %r15, %r15                       #../src/feautrier.cpp:184.29
  019c9 0f 88 c2 01 00 
        00               js ..B6.166 # Prob 0%                  #../src/feautrier.cpp:184.29
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.127:                       # Preds ..B6.126
                                # Execution count [2.33e+01]
  019cf 4d 3b 7a 10      cmpq 16(%r10), %r15                    #../src/feautrier.cpp:184.29
  019d3 0f 8d b8 01 00 
        00               jge ..B6.166 # Prob 0%                 #../src/feautrier.cpp:184.29
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.128:                       # Preds ..B6.127
                                # Execution count [2.32e+01]
  019d9 4c 0f af fe      imulq %rsi, %r15                       #../src/feautrier.cpp:184.29
  019dd 0f 28 c2         movaps %xmm2, %xmm0                    #../src/feautrier.cpp:184.48
  019e0 4d 03 fc         addq %r12, %r15                        #../src/feautrier.cpp:184.29
  019e3 f2 43 0f 58 44 
        e5 00            addsd (%r13,%r12,8), %xmm0             #../src/feautrier.cpp:184.48
  019ea f2 43 0f 10 4c 
        f9 08            movsd 8(%r9,%r15,8), %xmm1             #../src/feautrier.cpp:184.29
  019f1 f2 0f 5e c8      divsd %xmm0, %xmm1                     #../src/feautrier.cpp:184.48
  019f5 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:184.13
  019f8 0f 8c 93 01 00 
        00               jl ..B6.166 # Prob 0%                  #../src/feautrier.cpp:184.13
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.129:                       # Preds ..B6.128
                                # Execution count [2.48e+01]
  019fe 4d 3b e0         cmpq %r8, %r12                         #../src/feautrier.cpp:184.13
  01a01 0f 8d 8a 01 00 
        00               jge ..B6.166 # Prob 5%                 #../src/feautrier.cpp:184.13
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r13 r14 r15 xmm1 xmm2
..B6.130:                       # Preds ..B6.129
                                # Execution count [2.32e+01]
  01a07 f2 43 0f 11 0c 
        f9               movsd %xmm1, (%r9,%r15,8)              #../src/feautrier.cpp:184.13
  01a0d 49 89 fc         movq %rdi, %r12                        #../src/feautrier.cpp:182.34
  01a10 49 3b fb         cmpq %r11, %rdi                        #../src/feautrier.cpp:182.31
  01a13 7c 98            jl ..B6.124 # Prob 82%                 #../src/feautrier.cpp:182.31
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 xmm2
..B6.132:                       # Preds ..B6.130 ..B6.122
                                # Execution count [5.00e+00]
  01a15 0f               .byte 15                               #../src/feautrier.cpp:180.31
  01a16 1f               .byte 31                               #../src/feautrier.cpp:180.31
  01a17 44               .byte 68                               #../src/feautrier.cpp:180.31
  01a18 00               .byte 0                                #../src/feautrier.cpp:180.31
  01a19 00               .byte 0                                #../src/feautrier.cpp:180.31
  01a1a 48 ff c2         incq %rdx                              #../src/feautrier.cpp:180.31
  01a1d 48 ff c8         decq %rax                              #../src/feautrier.cpp:180.31
  01a20 48 3b d1         cmpq %rcx, %rdx                        #../src/feautrier.cpp:180.24
  01a23 0f 8c 78 ff ff 
        ff               jl ..B6.122 # Prob 82%                 #../src/feautrier.cpp:180.24
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r13 r14 xmm2
..B6.133:                       # Preds ..B6.132
                                # Execution count [9.00e-01]
  01a29 4c 8b 3c 24      movq (%rsp), %r15                      #[spill]
  01a2d ba 01 00 00 00   movl $1, %edx                          #../src/feautrier.cpp:191.15
  01a32 48 c7 c0 ff ff 
        ff ff            movq $-1, %rax                         #../src/feautrier.cpp:191.15
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r14 r15 xmm2
..B6.134:                       # Preds ..B6.144 ..B6.133
                                # Execution count [5.00e+00]
  01a39 49 89 d3         movq %rdx, %r11                        #../src/feautrier.cpp:193.19
  01a3c 48 3b d3         cmpq %rbx, %rdx                        #../src/feautrier.cpp:193.26
  01a3f 7d 68            jge ..B6.144 # Prob 10%                #../src/feautrier.cpp:193.26
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r14 r15 xmm2
..B6.136:                       # Preds ..B6.134 ..B6.142
                                # Execution count [2.50e+01]
  01a41 4d 89 dc         movq %r11, %r12                        #../src/feautrier.cpp:195.32
  01a44 4e 8d 2c 18      lea (%rax,%r11), %r13                  #../src/feautrier.cpp:195.29
  01a48 49 ff cc         decq %r12                              #../src/feautrier.cpp:195.32
  01a4b 0f 88 40 01 00 
        00               js ..B6.166 # Prob 0%                  #../src/feautrier.cpp:195.29
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.137:                       # Preds ..B6.136
                                # Execution count [2.48e+01]
  01a51 4d 3b e0         cmpq %r8, %r12                         #../src/feautrier.cpp:195.29
  01a54 0f 8d 37 01 00 
        00               jge ..B6.166 # Prob 5%                 #../src/feautrier.cpp:195.29
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r13 r14 r15 xmm2
..B6.138:                       # Preds ..B6.137
                                # Execution count [2.35e+01]
  01a5a 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:195.29
  01a5d 0f 88 2e 01 00 
        00               js ..B6.166 # Prob 0%                  #../src/feautrier.cpp:195.29
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r13 r14 r15 xmm2
..B6.139:                       # Preds ..B6.138
                                # Execution count [2.33e+01]
  01a63 4d 3b 6a 10      cmpq 16(%r10), %r13                    #../src/feautrier.cpp:195.29
  01a67 0f 8d 24 01 00 
        00               jge ..B6.166 # Prob 0%                 #../src/feautrier.cpp:195.29
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r13 r14 r15 xmm2
..B6.140:                       # Preds ..B6.139
                                # Execution count [2.32e+01]
  01a6d 4c 0f af ee      imulq %rsi, %r13                       #../src/feautrier.cpp:195.29
  01a71 0f 28 c2         movaps %xmm2, %xmm0                    #../src/feautrier.cpp:195.48
  01a74 4d 03 eb         addq %r11, %r13                        #../src/feautrier.cpp:195.29
  01a77 f2 42 0f 58 44 
        dd 00            addsd (%rbp,%r11,8), %xmm0             #../src/feautrier.cpp:195.48
  01a7e f2 43 0f 10 4c 
        e9 f8            movsd -8(%r9,%r13,8), %xmm1            #../src/feautrier.cpp:195.29
  01a85 f2 0f 5e c8      divsd %xmm0, %xmm1                     #../src/feautrier.cpp:195.48
  01a89 4d 85 db         testq %r11, %r11                       #../src/feautrier.cpp:195.13
  01a8c 0f 8c ff 00 00 
        00               jl ..B6.166 # Prob 0%                  #../src/feautrier.cpp:195.13
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r13 r14 r15 xmm1 xmm2
..B6.141:                       # Preds ..B6.140
                                # Execution count [2.48e+01]
  01a92 4d 3b d8         cmpq %r8, %r11                         #../src/feautrier.cpp:195.13
  01a95 0f 8d f6 00 00 
        00               jge ..B6.166 # Prob 5%                 #../src/feautrier.cpp:195.13
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r13 r14 r15 xmm1 xmm2
..B6.142:                       # Preds ..B6.141
                                # Execution count [2.32e+01]
  01a9b 49 ff c3         incq %r11                              #../src/feautrier.cpp:193.32
  01a9e f2 43 0f 11 0c 
        e9               movsd %xmm1, (%r9,%r13,8)              #../src/feautrier.cpp:195.13
  01aa4 4c 3b db         cmpq %rbx, %r11                        #../src/feautrier.cpp:193.26
  01aa7 7c 98            jl ..B6.136 # Prob 82%                 #../src/feautrier.cpp:193.26
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r14 r15 xmm2
..B6.144:                       # Preds ..B6.142 ..B6.134
                                # Execution count [5.00e+00]
  01aa9 0f               .byte 15                               #../src/feautrier.cpp:191.31
  01aaa 1f               .byte 31                               #../src/feautrier.cpp:191.31
  01aab 44               .byte 68                               #../src/feautrier.cpp:191.31
  01aac 00               .byte 0                                #../src/feautrier.cpp:191.31
  01aad 00               .byte 0                                #../src/feautrier.cpp:191.31
  01aae 48 ff c2         incq %rdx                              #../src/feautrier.cpp:191.31
  01ab1 48 ff c8         decq %rax                              #../src/feautrier.cpp:191.31
  01ab4 48 3b d1         cmpq %rcx, %rdx                        #../src/feautrier.cpp:191.24
  01ab7 7c 80            jl ..B6.134 # Prob 82%                 #../src/feautrier.cpp:191.24
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r14 r15 xmm2
..B6.146:                       # Preds ..B6.144 ..B6.120
                                # Execution count [1.00e+00]
  01ab9 4d 85 ff         testq %r15, %r15                       #../src/feautrier.cpp:202.13
  01abc 74 08            je ..B6.148 # Prob 32%                 #../src/feautrier.cpp:202.13
                                # LOE rbp r14 r15
..B6.147:                       # Preds ..B6.146
                                # Execution count [6.74e-01]
  01abe 4c 89 ff         movq %r15, %rdi                        #../src/feautrier.cpp:202.3
  01ac1 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:202.3
                                # LOE rbp r14
..B6.148:                       # Preds ..B6.147 ..B6.146
                                # Execution count [1.00e+00]
  01ac6 4d 85 f6         testq %r14, %r14                       #../src/feautrier.cpp:203.13
  01ac9 74 08            je ..B6.150 # Prob 32%                 #../src/feautrier.cpp:203.13
                                # LOE rbp r14
..B6.149:                       # Preds ..B6.148
                                # Execution count [6.74e-01]
  01acb 4c 89 f7         movq %r14, %rdi                        #../src/feautrier.cpp:203.3
  01ace e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:203.3
                                # LOE rbp
..B6.150:                       # Preds ..B6.149 ..B6.148
                                # Execution count [6.74e-01]
  01ad3 48 8b 7c 24 60   movq 96(%rsp), %rdi                    #../src/feautrier.cpp:204.3[spill]
  01ad8 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:204.3
                                # LOE rbp
..B6.151:                       # Preds ..B6.150
                                # Execution count [1.00e+00]
  01add 48 85 ed         testq %rbp, %rbp                       #../src/feautrier.cpp:205.13
  01ae0 74 08            je ..B6.153 # Prob 32%                 #../src/feautrier.cpp:205.13
                                # LOE rbp
..B6.152:                       # Preds ..B6.151
                                # Execution count [6.74e-01]
  01ae2 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:205.3
  01ae5 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:205.3
                                # LOE
..B6.153:                       # Preds ..B6.152 ..B6.151
                                # Execution count [1.00e+00]
  01aea 33 c0            xorl %eax, %eax                        #../src/feautrier.cpp:208.10
  01aec 48 83 c4 68      addq $104, %rsp                        #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 56
	.cfi_restore 6
  01af0 5d               popq %rbp                              #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 48
	.cfi_restore 3
  01af1 5b               popq %rbx                              #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 40
	.cfi_restore 15
  01af2 41 5f            popq %r15                              #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 32
	.cfi_restore 14
  01af4 41 5e            popq %r14                              #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 24
	.cfi_restore 13
  01af6 41 5d            popq %r13                              #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 16
	.cfi_restore 12
  01af8 41 5c            popq %r12                              #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 8
  01afa c3               ret                                    #../src/feautrier.cpp:208.10
	.cfi_def_cfa_offset 160
	.cfi_offset 3, -48
	.cfi_offset 6, -56
	.cfi_offset 12, -16
	.cfi_offset 13, -24
	.cfi_offset 14, -32
	.cfi_offset 15, -40
                                # LOE
..B6.154:                       # Preds ..B6.52
                                # Execution count [6.72e-02]: Infreq
  01afb 48 8d 0c dd 00 
        00 00 00         lea (,%rbx,8), %rcx                    #../src/feautrier.cpp:100.5
  01b03 49 8d 04 df      lea (%r15,%rbx,8), %rax                #../src/feautrier.cpp:100.5
  01b07 0f 84 98 fa ff 
        ff               je ..B6.84 # Prob 50%                  #../src/feautrier.cpp:98.13
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.155:                       # Preds ..B6.154
                                # Execution count [3.36e-02]: Infreq
  01b0d 48 83 7c 24 18 
        00               cmpq $0, 24(%rsp)                      #../src/feautrier.cpp:110.30[spill]
  01b13 0f 89 dc fa ff 
        ff               jns ..B6.86 # Prob 50%                 #../src/feautrier.cpp:110.30
  01b19 e9 c2 fb ff ff   jmp ..B6.97 # Prob 100%                #../src/feautrier.cpp:110.30
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.158:                       # Preds ..B6.93 ..B6.94
                                # Execution count [1.11e+00]: Infreq
  01b1e 4d 89 e9         movq %r13, %r9                         #../src/feautrier.cpp:115.3
  01b21 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:115.3
  01b24 49 c1 e9 3f      shrq $63, %r9                          #../src/feautrier.cpp:115.3
  01b28 ba 01 00 00 00   movl $1, %edx                          #../src/feautrier.cpp:115.3
  01b2d 4d 03 cd         addq %r13, %r9                         #../src/feautrier.cpp:25.5
  01b30 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:117.5
  01b33 49 d1 f9         sarq $1, %r9                           #../src/feautrier.cpp:25.5
  01b36 4d 85 c9         testq %r9, %r9                         #../src/feautrier.cpp:115.3
  01b39 76 2f            jbe ..B6.162 # Prob 10%                #../src/feautrier.cpp:115.3
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.159:                       # Preds ..B6.158
                                # Execution count [1.00e+00]: Infreq
  01b3b 48 8b 74 24 20   movq 32(%rsp), %rsi                    #../src/feautrier.cpp:117.5[spill]
  01b40 4b 8d 14 e0      lea (%r8,%r12,8), %rdx                 #../src/feautrier.cpp:117.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.160:                       # Preds ..B6.160 ..B6.159
                                # Execution count [2.78e+00]: Infreq
  01b44 49 8b 3c 33      movq (%r11,%rsi), %rdi                 #../src/feautrier.cpp:117.17
  01b48 49 ff c2         incq %r10                              #../src/feautrier.cpp:115.3
  01b4b 49 89 3c 13      movq %rdi, (%r11,%rdx)                 #../src/feautrier.cpp:117.5
  01b4f 49 8b 7c 33 08   movq 8(%r11,%rsi), %rdi                #../src/feautrier.cpp:117.17
  01b54 49 89 7c 13 08   movq %rdi, 8(%r11,%rdx)                #../src/feautrier.cpp:117.5
  01b59 49 83 c3 10      addq $16, %r11                         #../src/feautrier.cpp:115.3
  01b5d 4d 3b d1         cmpq %r9, %r10                         #../src/feautrier.cpp:115.3
  01b60 72 e2            jb ..B6.160 # Prob 64%                 #../src/feautrier.cpp:115.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B6.161:                       # Preds ..B6.160
                                # Execution count [1.00e+00]: Infreq
  01b62 4a 8d 14 55 01 
        00 00 00         lea 1(,%r10,2), %rdx                   #../src/feautrier.cpp:117.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.162:                       # Preds ..B6.161 ..B6.158
                                # Execution count [1.11e+00]: Infreq
  01b6a 48 ff ca         decq %rdx                              #../src/feautrier.cpp:117.5
  01b6d 49 3b d5         cmpq %r13, %rdx                        #../src/feautrier.cpp:115.3
  01b70 0f 83 6a fb ff 
        ff               jae ..B6.97 # Prob 10%                 #../src/feautrier.cpp:115.3
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.163:                       # Preds ..B6.162
                                # Execution count [1.00e+00]: Infreq
  01b76 48 8b 74 24 20   movq 32(%rsp), %rsi                    #../src/feautrier.cpp:117.17[spill]
  01b7b 4f 8d 14 e0      lea (%r8,%r12,8), %r10                 #../src/feautrier.cpp:117.5
  01b7f f2 0f 10 40 f8   movsd -8(%rax), %xmm0                  #../src/feautrier.cpp:142.28
  01b84 4c 8b 0c d6      movq (%rsi,%rdx,8), %r9                #../src/feautrier.cpp:117.17
  01b88 4d 89 0c d2      movq %r9, (%r10,%rdx,8)                #../src/feautrier.cpp:117.5
  01b8c e9 54 fb ff ff   jmp ..B6.98 # Prob 100%                #../src/feautrier.cpp:117.5
                                # LOE rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2
..B6.166:                       # Preds ..B6.106 ..B6.107 ..B6.113 ..B6.110 ..B6.111
                                #       ..B6.112 ..B6.116 ..B6.117 ..B6.118 ..B6.119
                                #       ..B6.127 ..B6.124 ..B6.125 ..B6.126 ..B6.129
                                #       ..B6.128 ..B6.139 ..B6.136 ..B6.137 ..B6.138
                                #       ..B6.140 ..B6.141
                                # Execution count [1.84e+00]: Infreq
  01b91 bf fe ff ff ff   movl $.L_2__STRING.6, %edi             #../src/feautrier.cpp:195.13
  01b96 be fe ff ff ff   movl $.L_2__STRING.7, %esi             #../src/feautrier.cpp:195.13
  01b9b ba 6d 01 00 00   movl $365, %edx                        #../src/feautrier.cpp:195.13
  01ba0 b9 00 00 00 00   movl $__$U5, %ecx                      #../src/feautrier.cpp:195.13
  01ba5 e8 fc ff ff ff  #       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #../src/feautrier.cpp:195.13
                                # LOE
..B6.189:                       # Preds ..B6.55 ..B6.57 ..B6.59
                                # Execution count [0.00e+00]: Infreq
  01baa 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:74.5
  01bad 49 83 fa 01      cmpq $1, %r10                          #../src/feautrier.cpp:74.5
  01bb1 0f 82 59 f9 ff 
        ff               jb ..B6.80 # Prob 50%                  #../src/feautrier.cpp:74.5
                                # LOE rbx rbp r8 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.190:                       # Preds ..B6.72 ..B6.189
                                # Execution count [0.00e+00]: Infreq
  01bb7 33 f6            xorl %esi, %esi                        #../src/feautrier.cpp:74.5
  01bb9 e9 fe f8 ff ff   jmp ..B6.76 # Prob 100%                #../src/feautrier.cpp:74.5
                                # LOE rbx rbp rsi r8 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B6.191:                       # Preds ..B6.53
                                # Execution count [6.05e-02]: Infreq
  01bbe 48 8d 0c dd 00 
        00 00 00         lea (,%rbx,8), %rcx                    #../src/feautrier.cpp:100.5
  01bc6 49 8d 04 df      lea (%r15,%rbx,8), %rax                #../src/feautrier.cpp:100.5
  01bca 0f 84 d5 f9 ff 
        ff               je ..B6.84 # Prob 50%                  #../src/feautrier.cpp:98.13
  01bd0 e9 20 fa ff ff   jmp ..B6.86 # Prob 100%                #../src/feautrier.cpp:98.13
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.192:                       # Preds ..B6.31 ..B6.29 ..B6.27
                                # Execution count [0.00e+00]: Infreq
  01bd5 33 f6            xorl %esi, %esi                        #../src/feautrier.cpp:64.5
  01bd7 48 83 f9 01      cmpq $1, %rcx                          #../src/feautrier.cpp:64.5
  01bdb 0f 82 e6 f5 ff 
        ff               jb ..B6.52 # Prob 50%                  #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm1 xmm2
..B6.193:                       # Preds ..B6.44 ..B6.192
                                # Execution count [0.00e+00]: Infreq
  01be1 33 ff            xorl %edi, %edi                        #../src/feautrier.cpp:64.5
  01be3 e9 82 f5 ff ff   jmp ..B6.48 # Prob 100%                #../src/feautrier.cpp:64.5
                                # LOE rdx rcx rbx rbp rsi rdi r8 r12 r13 r14 r15 xmm1 xmm2
..B6.194:                       # Preds ..B6.23
                                # Execution count [0.00e+00]: Infreq
  01be8 f2 0f 10 15 fc 
        ff ff ff         movsd .L_2il0floatpacket.62(%rip), %xmm2 #../src/feautrier.cpp:83.19
  01bf0 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:72.12
  01bf3 7e 17            jle ..B6.196 # Prob 16%                #../src/feautrier.cpp:72.12
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.195:                       # Preds ..B6.194
                                # Execution count [0.00e+00]: Infreq
  01bf5 f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:76.19
  01bfd 49 8d 44 24 ff   lea -1(%r12), %rax                     #../src/feautrier.cpp:110.15
  01c02 48 89 44 24 18   movq %rax, 24(%rsp)                    #../src/feautrier.cpp:110.15[spill]
  01c07 e9 ce f5 ff ff   jmp ..B6.54 # Prob 100%                #../src/feautrier.cpp:110.15
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.196:                       # Preds ..B6.194
                                # Execution count [3.23e-02]: Infreq
  01c0c 48 8d 0c dd 00 
        00 00 00         lea (,%rbx,8), %rcx                    #../src/feautrier.cpp:100.5
  01c14 49 8d 04 df      lea (%r15,%rbx,8), %rax                #../src/feautrier.cpp:100.5
  01c18 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:88.14
  01c1b 75 16            jne ..B6.198 # Prob 50%                #../src/feautrier.cpp:88.14
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B6.197:                       # Preds ..B6.196
                                # Execution count [1.62e-02]: Infreq
  01c1d f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:91.12
  01c25 48 c7 44 24 18 
        ff ff ff ff      movq $-1, 24(%rsp)                     #../src/feautrier.cpp:110.15[spill]
  01c2e e9 2c f9 ff ff   jmp ..B6.81 # Prob 100%                #../src/feautrier.cpp:110.15
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B6.198:                       # Preds ..B6.196
                                # Execution count [1.62e-02]: Infreq
  01c33 49 8d 54 24 ff   lea -1(%r12), %rdx                     #../src/feautrier.cpp:110.15
  01c38 48 89 54 24 18   movq %rdx, 24(%rsp)                    #../src/feautrier.cpp:110.15[spill]
  01c3d e9 56 f9 ff ff   jmp ..B6.82 # Prob 100%                #../src/feautrier.cpp:110.15
  01c42 0f 1f 80 00 00 
        00 00 0f 1f 80 
        00 00 00 00      .align    16,0x90
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
	.cfi_endproc
# mark_end;
	.type	_Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl,@function
	.size	_Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl,.-_Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl
	.data
# -- End  _Z9feautrierlPdS_lS_S_S_N5Eigen3RefINS0_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS0_11OuterStrideILin1EEEEEl
	.section .text._ZNSt11char_traitsIcE6lengthEPKc, "xaG",@progbits,_ZNSt11char_traitsIcE6lengthEPKc,comdat
..TXTST5:
# -- Begin  _ZNSt11char_traitsIcE6lengthEPKc
	.section .text._ZNSt11char_traitsIcE6lengthEPKc, "xaG",@progbits,_ZNSt11char_traitsIcE6lengthEPKc,comdat
# mark_begin;
       .align    16,0x90
	.weak _ZNSt11char_traitsIcE6lengthEPKc
# --- std::char_traits<char>::length(const std::char_traits<char>::char_type *)
_ZNSt11char_traitsIcE6lengthEPKc:
# parameter 1: %rdi
..B7.1:                         # Preds ..B7.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__ZNSt11char_traitsIcE6lengthEPKc.234:
..L235:
                                                        #/usr/include/c++/5/bits/char_traits.h:267.7
  00000 56               pushq %rsi                             #/usr/include/c++/5/bits/char_traits.h:267.7
	.cfi_def_cfa_offset 16
  00001 48 89 fa         movq %rdi, %rdx                        #/usr/include/c++/5/bits/char_traits.h:267.16
  00004 48 89 d1         movq %rdx, %rcx                        #/usr/include/c++/5/bits/char_traits.h:267.16
  00007 48 83 e2 f0      andq $-16, %rdx                        #/usr/include/c++/5/bits/char_traits.h:267.16
  0000b 66 0f ef c0      pxor %xmm0, %xmm0                      #/usr/include/c++/5/bits/char_traits.h:267.16
  0000f 66 0f 74 02      pcmpeqb (%rdx), %xmm0                  #/usr/include/c++/5/bits/char_traits.h:267.16
  00013 66 0f d7 c0      pmovmskb %xmm0, %eax                   #/usr/include/c++/5/bits/char_traits.h:267.16
  00017 83 e1 0f         andl $15, %ecx                         #/usr/include/c++/5/bits/char_traits.h:267.16
  0001a d3 e8            shrl %cl, %eax                         #/usr/include/c++/5/bits/char_traits.h:267.16
  0001c 0f bc c0         bsf %eax, %eax                         #/usr/include/c++/5/bits/char_traits.h:267.16
  0001f 75 0b            jne ..L237 # Prob 60%                  #/usr/include/c++/5/bits/char_traits.h:267.16
  00021 48 89 d0         movq %rdx, %rax                        #/usr/include/c++/5/bits/char_traits.h:267.16
  00024 48 03 d1         addq %rcx, %rdx                        #/usr/include/c++/5/bits/char_traits.h:267.16
  00027 e8 fc ff ff ff   call __intel_sse2_strlen               #/usr/include/c++/5/bits/char_traits.h:267.16
..L237:                                                         #
                                # LOE rax rbx rbp r12 r13 r14 r15
..B7.4:                         # Preds ..B7.1
                                # Execution count [1.00e+00]
  0002c 59               popq %rcx                              #/usr/include/c++/5/bits/char_traits.h:267.16
	.cfi_def_cfa_offset 8
  0002d c3               ret                                    #/usr/include/c++/5/bits/char_traits.h:267.16
  0002e 66 90            .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_ZNSt11char_traitsIcE6lengthEPKc,@function
	.size	_ZNSt11char_traitsIcE6lengthEPKc,.-_ZNSt11char_traitsIcE6lengthEPKc
	.data
# -- End  _ZNSt11char_traitsIcE6lengthEPKc
	.text
# -- Begin  _Z9feautrierlPdS_lS_S_S_
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z9feautrierlPdS_lS_S_S_
# --- feautrier(long, double *, double *, long, double *, double *, double *)
_Z9feautrierlPdS_lS_S_S_:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
# parameter 6: %r9
# parameter 7: 160 + %rsp
..B8.1:                         # Preds ..B8.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z9feautrierlPdS_lS_S_S_.240:
..L241:
                                                        #../src/feautrier.cpp:229.1
  01c50 41 54            pushq %r12                             #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
  01c52 41 55            pushq %r13                             #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
  01c54 41 56            pushq %r14                             #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
  01c56 41 57            pushq %r15                             #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
  01c58 53               pushq %rbx                             #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
  01c59 55               pushq %rbp                             #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
  01c5a 48 83 ec 68      subq $104, %rsp                        #../src/feautrier.cpp:229.1
	.cfi_def_cfa_offset 160
  01c5e 49 89 cc         movq %rcx, %r12                        #../src/feautrier.cpp:229.1
  01c61 49 89 fd         movq %rdi, %r13                        #../src/feautrier.cpp:229.1
  01c64 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:238.30
  01c6e 4c 89 4c 24 58   movq %r9, 88(%rsp)                     #../src/feautrier.cpp:229.1[spill]
  01c73 66 0f ef c0      pxor %xmm0, %xmm0                      #../src/feautrier.cpp:233.2
  01c77 4c 89 44 24 50   movq %r8, 80(%rsp)                     #../src/feautrier.cpp:229.1[spill]
  01c7c 48 89 54 24 48   movq %rdx, 72(%rsp)                    #../src/feautrier.cpp:229.1[spill]
  01c81 4b 8d 5c 25 00   lea (%r13,%r12), %rbx                  #../src/feautrier.cpp:231.20
  01c86 48 89 74 24 20   movq %rsi, 32(%rsp)                    #../src/feautrier.cpp:229.1[spill]
  01c8b f2 0f 11 44 24 
        30               movsd %xmm0, 48(%rsp)                  #../src/feautrier.cpp:233.2[spill]
  01c91 f2 0f 11 44 24 
        40               movsd %xmm0, 64(%rsp)                  #../src/feautrier.cpp:234.3[spill]
  01c97 f2 0f 11 44 24 
        28               movsd %xmm0, 40(%rsp)                  #../src/feautrier.cpp:235.3[spill]
  01c9d f2 0f 11 44 24 
        38               movsd %xmm0, 56(%rsp)                  #../src/feautrier.cpp:236.2[spill]
  01ca3 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:238.30
  01ca6 73 2e            jae ..B8.6 # Prob 50%                  #../src/feautrier.cpp:238.30
                                # LOE rbx r12 r13
..B8.2:                         # Preds ..B8.1 ..B8.6
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_.255:
  01ca8 e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:238.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.256:
                                # LOE rbx r12 r13
..B8.3:                         # Preds ..B8.2
                                # Execution count [5.25e-01]
  01cad 48 8d 2c dd 00 
        00 00 00         lea (,%rbx,8), %rbp                    #../src/feautrier.cpp:238.30
  01cb5 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:238.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.257:
  01cb8 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:238.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.258:
                                # LOE rax rbx rbp r12 r13
..B8.133:                       # Preds ..B8.3
                                # Execution count [5.25e-01]
  01cbd 49 89 c7         movq %rax, %r15                        #../src/feautrier.cpp:238.30
                                # LOE rbx rbp r12 r13 r15
..B8.4:                         # Preds ..B8.133
                                # Execution count [5.25e-01]
  01cc0 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:239.30
  01cca 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:239.30
  01ccd 72 2e            jb ..B8.9 # Prob 50%                   #../src/feautrier.cpp:239.30
                                # LOE rbx rbp r12 r13 r15
..B8.5:                         # Preds ..B8.4
                                # Execution count [2.63e-01]
  01ccf 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:239.30
  01cd2 78 29            js ..B8.9 # Prob 5%                    #../src/feautrier.cpp:239.30
  01cd4 eb 4d            jmp ..B8.13 # Prob 100%                #../src/feautrier.cpp:239.30
                                # LOE rbx rbp r12 r13 r15
..B8.6:                         # Preds ..B8.1
                                # Execution count [5.00e-01]
  01cd6 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:238.30
  01cd9 78 cd            js ..B8.2 # Prob 5%                    #../src/feautrier.cpp:238.30
                                # LOE rbx r12 r13
..B8.7:                         # Preds ..B8.6
                                # Execution count [4.75e-01]
  01cdb 48 8d 2c dd 00 
        00 00 00         lea (,%rbx,8), %rbp                    #../src/feautrier.cpp:238.30
  01ce3 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:238.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.259:
  01ce6 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:238.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.260:
                                # LOE rax rbx rbp r12 r13
..B8.134:                       # Preds ..B8.7
                                # Execution count [4.75e-01]
  01ceb 49 89 c7         movq %rax, %r15                        #../src/feautrier.cpp:238.30
                                # LOE rbx rbp r12 r13 r15
..B8.8:                         # Preds ..B8.134
                                # Execution count [4.75e-01]
  01cee 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:239.30
  01cf8 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:239.30
  01cfb 73 26            jae ..B8.13 # Prob 50%                 #../src/feautrier.cpp:239.30
                                # LOE rbx rbp r12 r13 r15
..B8.9:                         # Preds ..B8.8 ..B8.5 ..B8.4
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_.261:
  01cfd e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:239.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.262:
                                # LOE rbx rbp r12 r13 r15
..B8.10:                        # Preds ..B8.9
                                # Execution count [5.25e-01]
  01d02 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:239.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.263:
  01d05 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:239.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.264:
                                # LOE rax rbx rbp r12 r13 r15
..B8.135:                       # Preds ..B8.10
                                # Execution count [5.25e-01]
  01d0a 49 89 c6         movq %rax, %r14                        #../src/feautrier.cpp:239.30
                                # LOE rbx rbp r12 r13 r14 r15
..B8.11:                        # Preds ..B8.135
                                # Execution count [5.25e-01]
  01d0d 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:240.30
  01d17 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:240.30
  01d1a 72 21            jb ..B8.15 # Prob 50%                  #../src/feautrier.cpp:240.30
                                # LOE rbx rbp r12 r13 r14 r15
..B8.12:                        # Preds ..B8.11
                                # Execution count [2.63e-01]
  01d1c 48 85 db         testq %rbx, %rbx                       #../src/feautrier.cpp:240.30
  01d1f 78 1c            js ..B8.15 # Prob 5%                   #../src/feautrier.cpp:240.30
  01d21 eb 1f            jmp ..B8.16 # Prob 100%                #../src/feautrier.cpp:240.30
                                # LOE rbx rbp r12 r13 r14 r15
..B8.13:                        # Preds ..B8.8 ..B8.5
                                # Execution count [4.75e-01]
  01d23 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:239.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.265:
  01d26 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:239.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.266:
                                # LOE rax rbx rbp r12 r13 r15
..B8.136:                       # Preds ..B8.13
                                # Execution count [4.75e-01]
  01d2b 49 89 c6         movq %rax, %r14                        #../src/feautrier.cpp:239.30
                                # LOE rbx rbp r12 r13 r14 r15
..B8.14:                        # Preds ..B8.136
                                # Execution count [4.75e-01]
  01d2e 48 b8 ff ff ff 
        ff ff ff ff 0f   movq $0xfffffffffffffff, %rax          #../src/feautrier.cpp:240.30
  01d38 48 3b c3         cmpq %rbx, %rax                        #../src/feautrier.cpp:240.30
  01d3b 73 05            jae ..B8.16 # Prob 50%                 #../src/feautrier.cpp:240.30
                                # LOE rbx rbp r12 r13 r14 r15
..B8.15:                        # Preds ..B8.14 ..B8.12 ..B8.11
                                # Execution count [5.25e-01]
..___tag_value__Z9feautrierlPdS_lS_S_S_.267:
  01d3d e8 fc ff ff ff  #       __cxa_throw_bad_array_new_length()
        call      __cxa_throw_bad_array_new_length              #../src/feautrier.cpp:240.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.268:
                                # LOE rbx rbp r12 r13 r14 r15
..B8.16:                        # Preds ..B8.14 ..B8.12 ..B8.15
                                # Execution count [1.00e+00]
  01d42 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:240.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.269:
  01d45 e8 fc ff ff ff  #       operator new[](std::size_t)
        call      _Znam                                         #../src/feautrier.cpp:240.30
..___tag_value__Z9feautrierlPdS_lS_S_S_.270:
                                # LOE rax rbx r12 r13 r14 r15
..B8.137:                       # Preds ..B8.16
                                # Execution count [1.00e+00]
  01d4a 48 89 c5         movq %rax, %rbp                        #../src/feautrier.cpp:240.30
                                # LOE rbx rbp r12 r13 r14 r15
..B8.17:                        # Preds ..B8.137
                                # Execution count [1.00e+00]
  01d4d 4c 8b 84 24 a0 
        00 00 00         movq 160(%rsp), %r8                    #../src/feautrier.cpp:229.1
  01d55 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:249.15
  01d58 0f 8e d6 0c 00 
        00               jle ..B8.126 # Prob 16%                #../src/feautrier.cpp:249.15
                                # LOE rbx rbp r8 r12 r13 r14 r15
..B8.18:                        # Preds ..B8.17
                                # Execution count [8.40e-01]
  01d5e f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:251.17
  01d66 4a 8d 0c e5 00 
        00 00 00         lea (,%r12,8), %rcx                    #../src/feautrier.cpp:251.5
  01d6e 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:249.28
  01d71 7e 39            jle ..B8.20 # Prob 16%                 #../src/feautrier.cpp:249.28
                                # LOE rcx rbx rbp r8 r12 r13 r14 r15 xmm1
..B8.19:                        # Preds ..B8.18
                                # Execution count [7.06e-01]
  01d73 48 8b 44 24 58   movq 88(%rsp), %rax                    #../src/feautrier.cpp:251.24[spill]
  01d78 0f 28 e9         movaps %xmm1, %xmm5                    #../src/feautrier.cpp:251.37
  01d7b 48 8b 54 24 48   movq 72(%rsp), %rdx                    #../src/feautrier.cpp:251.37[spill]
  01d80 f2 0f 10 10      movsd (%rax), %xmm2                    #../src/feautrier.cpp:251.24
  01d84 f2 0f 10 22      movsd (%rdx), %xmm4                    #../src/feautrier.cpp:251.37
  01d88 0f 28 c2         movaps %xmm2, %xmm0                    #../src/feautrier.cpp:251.37
  01d8b f2 0f 58 c4      addsd %xmm4, %xmm0                     #../src/feautrier.cpp:251.37
  01d8f f2 0f 5e e8      divsd %xmm0, %xmm5                     #../src/feautrier.cpp:251.37
  01d93 0f 28 dd         movaps %xmm5, %xmm3                    #../src/feautrier.cpp:251.50
  01d96 f2 0f 5e da      divsd %xmm2, %xmm3                     #../src/feautrier.cpp:251.50
  01d9a f2 0f 5e ec      divsd %xmm4, %xmm5                     #../src/feautrier.cpp:252.50
  01d9e f2 41 0f 11 5c 
        0f f8            movsd %xmm3, -8(%r15,%rcx)             #../src/feautrier.cpp:251.5
  01da5 f2 41 0f 11 6c 
        0e f8            movsd %xmm5, -8(%r14,%rcx)             #../src/feautrier.cpp:252.5
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1
..B8.20:                        # Preds ..B8.18 ..B8.19
                                # Execution count [8.40e-01]
  01dac 48 8b 54 24 58   movq 88(%rsp), %rdx                    #../src/feautrier.cpp:259.32[spill]
  01db1 0f 28 d9         movaps %xmm1, %xmm3                    #../src/feautrier.cpp:259.16
  01db4 49 8d 44 24 ff   lea -1(%r12), %rax                     #../src/feautrier.cpp:264.17
  01db9 48 89 44 24 18   movq %rax, 24(%rsp)                    #../src/feautrier.cpp:264.17[spill]
  01dbe 66 0f ef d2      pxor %xmm2, %xmm2                      #../src/feautrier.cpp:258.5
  01dc2 f2 41 0f 11 17   movsd %xmm2, (%r15)                    #../src/feautrier.cpp:258.5
  01dc7 4a 8d 14 e2      lea (%rdx,%r12,8), %rdx                #../src/feautrier.cpp:259.32
  01dcb f2 0f 10 42 f8   movsd -8(%rdx), %xmm0                  #../src/feautrier.cpp:259.16
  01dd0 f2 0f 5e d8      divsd %xmm0, %xmm3                     #../src/feautrier.cpp:259.16
  01dd4 0f 28 e3         movaps %xmm3, %xmm4                    #../src/feautrier.cpp:259.32
  01dd7 f2 0f 5e e0      divsd %xmm0, %xmm4                     #../src/feautrier.cpp:259.32
  01ddb f2 0f 10 15 fc 
        ff ff ff         movsd .L_2il0floatpacket.62(%rip), %xmm2 #../src/feautrier.cpp:261.5
  01de3 f2 41 0f 11 26   movsd %xmm4, (%r14)                    #../src/feautrier.cpp:259.5
  01de8 f2 0f 58 da      addsd %xmm2, %xmm3                     #../src/feautrier.cpp:261.27
  01dec f2 0f 11 5c 24 
        40               movsd %xmm3, 64(%rsp)                  #../src/feautrier.cpp:261.27[spill]
  01df2 f2 0f 58 e3      addsd %xmm3, %xmm4                     #../src/feautrier.cpp:261.65
  01df6 f2 0f 11 64 24 
        30               movsd %xmm4, 48(%rsp)                  #../src/feautrier.cpp:261.65[spill]
  01dfc 48 83 f8 01      cmpq $1, %rax                          #../src/feautrier.cpp:264.31
  01e00 0f 8e 71 04 00 
        00               jle ..B8.46 # Prob 50%                 #../src/feautrier.cpp:264.31
                                # LOE rdx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.21:                        # Preds ..B8.20
                                # Execution count [7.56e-01]
  01e06 49 8d 4c 24 fe   lea -2(%r12), %rcx                     #../src/feautrier.cpp:264.5
  01e0b 48 83 f9 08      cmpq $8, %rcx                          #../src/feautrier.cpp:264.5
  01e0f 0f 8c 0c 0c 00 
        00               jl ..B8.124 # Prob 10%                 #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.22:                        # Preds ..B8.21
                                # Execution count [7.56e-01]
  01e15 49 8d 46 08      lea 8(%r14), %rax                      #../src/feautrier.cpp:267.7
  01e19 48 83 e0 0f      andq $15, %rax                         #../src/feautrier.cpp:264.5
  01e1d 74 11            je ..B8.25 # Prob 50%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.23:                        # Preds ..B8.22
                                # Execution count [7.56e-01]
  01e1f 48 a9 07 00 00 
        00               testq $7, %rax                         #../src/feautrier.cpp:264.5
  01e25 0f 85 f6 0b 00 
        00               jne ..B8.124 # Prob 10%                #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.24:                        # Preds ..B8.23
                                # Execution count [3.78e-01]
  01e2b b8 01 00 00 00   movl $1, %eax                          #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.25:                        # Preds ..B8.24 ..B8.22
                                # Execution count [7.56e-01]
  01e30 48 8d 70 08      lea 8(%rax), %rsi                      #../src/feautrier.cpp:264.5
  01e34 48 3b ce         cmpq %rsi, %rcx                        #../src/feautrier.cpp:264.5
  01e37 0f 8c e4 0b 00 
        00               jl ..B8.124 # Prob 10%                 #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.26:                        # Preds ..B8.25
                                # Execution count [8.40e-01]
  01e3d 48 89 ce         movq %rcx, %rsi                        #../src/feautrier.cpp:264.5
  01e40 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:264.5
  01e43 48 2b f0         subq %rax, %rsi                        #../src/feautrier.cpp:264.5
  01e46 45 33 c9         xorl %r9d, %r9d                        #../src/feautrier.cpp:266.7
  01e49 48 83 e6 07      andq $7, %rsi                          #../src/feautrier.cpp:264.5
  01e4d 48 f7 de         negq %rsi                              #../src/feautrier.cpp:264.5
  01e50 48 03 f1         addq %rcx, %rsi                        #../src/feautrier.cpp:264.5
  01e53 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:264.5
  01e56 76 40            jbe ..B8.30 # Prob 9%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.28:                        # Preds ..B8.26 ..B8.28
                                # Execution count [4.20e+00]
  01e58 f2 42 0f 10 5c 
        ca f0            movsd -16(%rdx,%r9,8), %xmm3           #../src/feautrier.cpp:266.57
  01e5f 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:266.41
  01e62 f2 42 0f 10 6c 
        ca e8            movsd -24(%rdx,%r9,8), %xmm5           #../src/feautrier.cpp:266.41
  01e69 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:266.41
  01e6c 49 ff c9         decq %r9                               #../src/feautrier.cpp:264.5
  01e6f f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:266.41
  01e73 f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:266.41
  01e77 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:266.57
  01e7a f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:266.57
  01e7e f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:267.57
  01e82 f2 43 0f 11 64 
        d7 08            movsd %xmm4, 8(%r15,%r10,8)            #../src/feautrier.cpp:266.7
  01e89 f2 43 0f 11 74 
        d6 08            movsd %xmm6, 8(%r14,%r10,8)            #../src/feautrier.cpp:267.7
  01e90 49 ff c2         incq %r10                              #../src/feautrier.cpp:264.5
  01e93 4c 3b d0         cmpq %rax, %r10                        #../src/feautrier.cpp:264.5
  01e96 72 c0            jb ..B8.28 # Prob 82%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.30:                        # Preds ..B8.28 ..B8.26
                                # Execution count [0.00e+00]
  01e98 0f 10 05 fc ff 
        ff ff            movups .L_2il0floatpacket.61(%rip), %xmm0 #../src/feautrier.cpp:266.19
  01e9f 4c 8d 0c c5 00 
        00 00 00         lea (,%rax,8), %r9                     #../src/feautrier.cpp:264.5
  01ea7 49 f7 d9         negq %r9                               #../src/feautrier.cpp:264.5
  01eaa 4d 8d 54 c7 08   lea 8(%r15,%rax,8), %r10               #../src/feautrier.cpp:266.7
  01eaf 4c 89 0c 24      movq %r9, (%rsp)                       #../src/feautrier.cpp:264.5[spill]
  01eb3 49 f7 c2 0f 00 
        00 00            testq $15, %r10                        #../src/feautrier.cpp:264.5
  01eba 0f 84 52 01 00 
        00               je ..B8.34 # Prob 60%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.31:                        # Preds ..B8.30
                                # Execution count [7.56e-01]
  01ec0 4c 8b 4c 24 58   movq 88(%rsp), %r9                     #../src/feautrier.cpp:258.5[spill]
  01ec5 4c 89 64 24 10   movq %r12, 16(%rsp)                    #../src/feautrier.cpp:258.5[spill]
  01eca 4c 89 6c 24 08   movq %r13, 8(%rsp)                     #../src/feautrier.cpp:258.5[spill]
  01ecf 4f 8d 5c e1 e8   lea -24(%r9,%r12,8), %r11              #../src/feautrier.cpp:258.5
  01ed4 4f 8d 54 e1 d8   lea -40(%r9,%r12,8), %r10              #../src/feautrier.cpp:258.5
  01ed9 4b 8d 7c e1 c8   lea -56(%r9,%r12,8), %rdi              #../src/feautrier.cpp:258.5
  01ede 4f 8d 4c e1 b8   lea -72(%r9,%r12,8), %r9               #../src/feautrier.cpp:258.5
  01ee3 4c 8b 24 24      movq (%rsp), %r12                      #../src/feautrier.cpp:258.5[spill]
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B8.32:                        # Preds ..B8.31 ..B8.32
                                # Execution count [4.20e+00]
  01ee7 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:266.41
  01eea 4f 8d 2c 23      lea (%r11,%r12), %r13                  #../src/feautrier.cpp:266.41
  01eee f2 41 0f 10 65 
        08               movsd 8(%r13), %xmm4                   #../src/feautrier.cpp:266.26
  01ef4 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:266.41
  01ef8 66 41 0f 16 65 
        00               movhpd (%r13), %xmm4                   #../src/feautrier.cpp:266.26
  01efe f2 41 0f 10 75 
        00               movsd (%r13), %xmm6                    #../src/feautrier.cpp:266.26
  01f04 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:266.41
  01f07 66 41 0f 16 75 
        f8               movhpd -8(%r13), %xmm6                 #../src/feautrier.cpp:266.26
  01f0d 4f 8d 2c 22      lea (%r10,%r12), %r13                  #../src/feautrier.cpp:266.41
  01f11 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:266.41
  01f15 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:266.41
  01f19 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:266.57
  01f1c 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:266.57
  01f20 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:267.57
  01f24 41 0f 11 6c c7 
        08               movups %xmm5, 8(%r15,%rax,8)           #../src/feautrier.cpp:266.7
  01f2a 41 0f 11 7c c6 
        08               movups %xmm7, 8(%r14,%rax,8)           #../src/feautrier.cpp:267.7
  01f30 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:266.41
  01f33 f2 45 0f 10 4d 
        08               movsd 8(%r13), %xmm9                   #../src/feautrier.cpp:266.26
  01f39 66 45 0f 16 4d 
        00               movhpd (%r13), %xmm9                   #../src/feautrier.cpp:266.26
  01f3f f2 45 0f 10 5d 
        00               movsd (%r13), %xmm11                   #../src/feautrier.cpp:266.26
  01f45 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:266.41
  01f49 66 45 0f 16 5d 
        f8               movhpd -8(%r13), %xmm11                #../src/feautrier.cpp:266.26
  01f4f 4e 8d 2c 27      lea (%rdi,%r12), %r13                  #../src/feautrier.cpp:266.41
  01f53 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:266.41
  01f58 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:266.41
  01f5d 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:266.57
  01f61 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:266.57
  01f66 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:267.57
  01f6b 45 0f 11 54 c7 
        18               movups %xmm10, 24(%r15,%rax,8)         #../src/feautrier.cpp:266.7
  01f71 45 0f 11 64 c6 
        18               movups %xmm12, 24(%r14,%rax,8)         #../src/feautrier.cpp:267.7
  01f77 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:266.41
  01f7b f2 45 0f 10 75 
        08               movsd 8(%r13), %xmm14                  #../src/feautrier.cpp:266.26
  01f81 66 45 0f 16 75 
        00               movhpd (%r13), %xmm14                  #../src/feautrier.cpp:266.26
  01f87 f2 41 0f 10 5d 
        00               movsd (%r13), %xmm3                    #../src/feautrier.cpp:266.26
  01f8d 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:266.41
  01f91 66 41 0f 16 5d 
        f8               movhpd -8(%r13), %xmm3                 #../src/feautrier.cpp:266.26
  01f97 4f 8d 2c 21      lea (%r9,%r12), %r13                   #../src/feautrier.cpp:266.41
  01f9b 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:266.41
  01fa0 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:266.41
  01fa5 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:266.57
  01fa9 49 83 c4 c0      addq $-64, %r12                        #../src/feautrier.cpp:264.5
  01fad 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:266.57
  01fb2 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:267.57
  01fb6 45 0f 11 7c c7 
        28               movups %xmm15, 40(%r15,%rax,8)         #../src/feautrier.cpp:266.7
  01fbc 41 0f 11 64 c6 
        28               movups %xmm4, 40(%r14,%rax,8)          #../src/feautrier.cpp:267.7
  01fc2 f2 41 0f 10 75 
        08               movsd 8(%r13), %xmm6                   #../src/feautrier.cpp:266.26
  01fc8 66 41 0f 16 75 
        00               movhpd (%r13), %xmm6                   #../src/feautrier.cpp:266.26
  01fce f2 45 0f 10 45 
        00               movsd (%r13), %xmm8                    #../src/feautrier.cpp:266.26
  01fd4 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:266.41
  01fd7 66 45 0f 16 45 
        f8               movhpd -8(%r13), %xmm8                 #../src/feautrier.cpp:266.26
  01fdd 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:266.41
  01fe2 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:266.41
  01fe7 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:266.57
  01feb 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:266.57
  01fef 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:267.57
  01ff4 41 0f 11 7c c7 
        38               movups %xmm7, 56(%r15,%rax,8)          #../src/feautrier.cpp:266.7
  01ffa 45 0f 11 4c c6 
        38               movups %xmm9, 56(%r14,%rax,8)          #../src/feautrier.cpp:267.7
  02000 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:264.5
  02004 48 3b c6         cmpq %rsi, %rax                        #../src/feautrier.cpp:264.5
  02007 0f 82 da fe ff 
        ff               jb ..B8.32 # Prob 82%                  #../src/feautrier.cpp:264.5
  0200d e9 4d 01 00 00   jmp ..B8.36 # Prob 100%                #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B8.34:                        # Preds ..B8.30
                                # Execution count [7.56e-01]
  02012 4c 8b 5c 24 58   movq 88(%rsp), %r11                    #../src/feautrier.cpp:258.5[spill]
  02017 4c 89 64 24 10   movq %r12, 16(%rsp)                    #../src/feautrier.cpp:258.5[spill]
  0201c 4c 89 6c 24 08   movq %r13, 8(%rsp)                     #../src/feautrier.cpp:258.5[spill]
  02021 4f 8d 54 e3 e8   lea -24(%r11,%r12,8), %r10             #../src/feautrier.cpp:258.5
  02026 4f 8d 4c e3 d8   lea -40(%r11,%r12,8), %r9              #../src/feautrier.cpp:258.5
  0202b 4b 8d 7c e3 c8   lea -56(%r11,%r12,8), %rdi             #../src/feautrier.cpp:258.5
  02030 4f 8d 5c e3 b8   lea -72(%r11,%r12,8), %r11             #../src/feautrier.cpp:258.5
  02035 4c 8b 24 24      movq (%rsp), %r12                      #../src/feautrier.cpp:258.5[spill]
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B8.35:                        # Preds ..B8.34 ..B8.35
                                # Execution count [4.20e+00]
  02039 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:266.41
  0203c 4f 8d 2c 22      lea (%r10,%r12), %r13                  #../src/feautrier.cpp:266.41
  02040 f2 41 0f 10 65 
        08               movsd 8(%r13), %xmm4                   #../src/feautrier.cpp:266.26
  02046 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:266.41
  0204a 66 41 0f 16 65 
        00               movhpd (%r13), %xmm4                   #../src/feautrier.cpp:266.26
  02050 f2 41 0f 10 75 
        00               movsd (%r13), %xmm6                    #../src/feautrier.cpp:266.26
  02056 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:266.41
  02059 66 41 0f 16 75 
        f8               movhpd -8(%r13), %xmm6                 #../src/feautrier.cpp:266.26
  0205f 4f 8d 2c 21      lea (%r9,%r12), %r13                   #../src/feautrier.cpp:266.41
  02063 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:266.41
  02067 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:266.41
  0206b 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:266.57
  0206e 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:266.57
  02072 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:267.57
  02076 41 0f 11 6c c7 
        08               movups %xmm5, 8(%r15,%rax,8)           #../src/feautrier.cpp:266.7
  0207c 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:266.41
  0207f 41 0f 11 7c c6 
        08               movups %xmm7, 8(%r14,%rax,8)           #../src/feautrier.cpp:267.7
  02085 f2 45 0f 10 4d 
        08               movsd 8(%r13), %xmm9                   #../src/feautrier.cpp:266.26
  0208b 66 45 0f 16 4d 
        00               movhpd (%r13), %xmm9                   #../src/feautrier.cpp:266.26
  02091 f2 45 0f 10 5d 
        00               movsd (%r13), %xmm11                   #../src/feautrier.cpp:266.26
  02097 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:266.41
  0209b 66 45 0f 16 5d 
        f8               movhpd -8(%r13), %xmm11                #../src/feautrier.cpp:266.26
  020a1 4e 8d 2c 27      lea (%rdi,%r12), %r13                  #../src/feautrier.cpp:266.41
  020a5 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:266.41
  020aa 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:266.41
  020af 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:266.57
  020b3 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:266.57
  020b8 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:267.57
  020bd 45 0f 11 54 c7 
        18               movups %xmm10, 24(%r15,%rax,8)         #../src/feautrier.cpp:266.7
  020c3 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:266.41
  020c7 45 0f 11 64 c6 
        18               movups %xmm12, 24(%r14,%rax,8)         #../src/feautrier.cpp:267.7
  020cd f2 45 0f 10 75 
        08               movsd 8(%r13), %xmm14                  #../src/feautrier.cpp:266.26
  020d3 66 45 0f 16 75 
        00               movhpd (%r13), %xmm14                  #../src/feautrier.cpp:266.26
  020d9 f2 41 0f 10 5d 
        00               movsd (%r13), %xmm3                    #../src/feautrier.cpp:266.26
  020df 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:266.41
  020e3 66 41 0f 16 5d 
        f8               movhpd -8(%r13), %xmm3                 #../src/feautrier.cpp:266.26
  020e9 4f 8d 2c 23      lea (%r11,%r12), %r13                  #../src/feautrier.cpp:266.41
  020ed 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:266.41
  020f2 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:266.41
  020f7 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:266.57
  020fb 49 83 c4 c0      addq $-64, %r12                        #../src/feautrier.cpp:264.5
  020ff 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:266.57
  02104 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:267.57
  02108 45 0f 11 7c c7 
        28               movups %xmm15, 40(%r15,%rax,8)         #../src/feautrier.cpp:266.7
  0210e 41 0f 11 64 c6 
        28               movups %xmm4, 40(%r14,%rax,8)          #../src/feautrier.cpp:267.7
  02114 f2 41 0f 10 75 
        08               movsd 8(%r13), %xmm6                   #../src/feautrier.cpp:266.26
  0211a 66 41 0f 16 75 
        00               movhpd (%r13), %xmm6                   #../src/feautrier.cpp:266.26
  02120 f2 45 0f 10 45 
        00               movsd (%r13), %xmm8                    #../src/feautrier.cpp:266.26
  02126 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:266.41
  02129 66 45 0f 16 45 
        f8               movhpd -8(%r13), %xmm8                 #../src/feautrier.cpp:266.26
  0212f 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:266.41
  02134 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:266.41
  02139 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:266.57
  0213d 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:266.57
  02141 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:267.57
  02146 41 0f 11 7c c7 
        38               movups %xmm7, 56(%r15,%rax,8)          #../src/feautrier.cpp:266.7
  0214c 45 0f 11 4c c6 
        38               movups %xmm9, 56(%r14,%rax,8)          #../src/feautrier.cpp:267.7
  02152 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:264.5
  02156 48 3b c6         cmpq %rsi, %rax                        #../src/feautrier.cpp:264.5
  02159 0f 82 da fe ff 
        ff               jb ..B8.35 # Prob 82%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r15 xmm0 xmm1 xmm2
..B8.36:                        # Preds ..B8.32 ..B8.35
                                # Execution count [7.56e-01]
  0215f 4c 8b 6c 24 08   movq 8(%rsp), %r13                     #[spill]
  02164 4c 8b 64 24 10   movq 16(%rsp), %r12                    #[spill]
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.37:                        # Preds ..B8.36
                                # Execution count [7.56e-01]
  02169 48 8d 46 01      lea 1(%rsi), %rax                      #../src/feautrier.cpp:264.5
  0216d 48 3b c1         cmpq %rcx, %rax                        #../src/feautrier.cpp:264.5
  02170 0f 87 0b 01 00 
        00               ja ..B8.47 # Prob 50%                  #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.38:                        # Preds ..B8.37
                                # Execution count [7.56e-01]
  02176 48 2b ce         subq %rsi, %rcx                        #../src/feautrier.cpp:264.5
  02179 48 83 f9 02      cmpq $2, %rcx                          #../src/feautrier.cpp:264.5
  0217d 0f 8c aa 08 00 
        00               jl ..B8.125 # Prob 10%                 #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.39:                        # Preds ..B8.38
                                # Execution count [7.56e-01]
  02183 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:264.5
  02186 48 8d 04 f5 00 
        00 00 00         lea (,%rsi,8), %rax                    #../src/feautrier.cpp:266.41
  0218e 48 f7 d8         negq %rax                              #../src/feautrier.cpp:266.41
  02191 48 89 cf         movq %rcx, %rdi                        #../src/feautrier.cpp:264.5
  02194 48 83 e7 fe      andq $-2, %rdi                         #../src/feautrier.cpp:264.5
  02198 4c 89 6c 24 08   movq %r13, 8(%rsp)                     #../src/feautrier.cpp:258.5[spill]
  0219d 4d 8d 0c f7      lea (%r15,%rsi,8), %r9                 #../src/feautrier.cpp:266.7
  021a1 4c 89 64 24 10   movq %r12, 16(%rsp)                    #../src/feautrier.cpp:258.5[spill]
  021a6 4d 8d 14 f6      lea (%r14,%rsi,8), %r10                #../src/feautrier.cpp:267.7
  021aa 4d 89 dd         movq %r11, %r13                        #../src/feautrier.cpp:258.5
  021ad 48 8d 44 02 e8   lea -24(%rdx,%rax), %rax               #../src/feautrier.cpp:258.5
  021b2 0f 1f 80 00 00 
        00 00 0f 1f 80 
        00 00 00 00      .align    16,0x90
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r13 r14 r15 xmm0 xmm1 xmm2
..B8.40:                        # Preds ..B8.40 ..B8.39
                                # Execution count [4.20e+00]
  021c0 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:266.41
  021c3 4e 8d 24 28      lea (%rax,%r13), %r12                  #../src/feautrier.cpp:266.41
  021c7 f2 41 0f 10 64 
        24 08            movsd 8(%r12), %xmm4                   #../src/feautrier.cpp:266.26
  021ce 49 83 c5 f0      addq $-16, %r13                        #../src/feautrier.cpp:264.5
  021d2 66 41 0f 16 24 
        24               movhpd (%r12), %xmm4                   #../src/feautrier.cpp:266.26
  021d8 f2 41 0f 10 34 
        24               movsd (%r12), %xmm6                    #../src/feautrier.cpp:266.26
  021de 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:266.41
  021e1 66 41 0f 16 74 
        24 f8            movhpd -8(%r12), %xmm6                 #../src/feautrier.cpp:266.26
  021e8 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:266.41
  021ec 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:266.41
  021f0 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:266.57
  021f3 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:266.57
  021f7 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:267.57
  021fb 43 0f 11 6c d9 
        08               movups %xmm5, 8(%r9,%r11,8)            #../src/feautrier.cpp:266.7
  02201 43 0f 11 7c da 
        08               movups %xmm7, 8(%r10,%r11,8)           #../src/feautrier.cpp:267.7
  02207 49 83 c3 02      addq $2, %r11                          #../src/feautrier.cpp:264.5
  0220b 4c 3b df         cmpq %rdi, %r11                        #../src/feautrier.cpp:264.5
  0220e 72 b0            jb ..B8.40 # Prob 82%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r13 r14 r15 xmm0 xmm1 xmm2
..B8.41:                        # Preds ..B8.40
                                # Execution count [7.56e-01]
  02210 4c 8b 6c 24 08   movq 8(%rsp), %r13                     #[spill]
  02215 4c 8b 64 24 10   movq 16(%rsp), %r12                    #[spill]
                                # LOE rdx rcx rbx rbp rsi rdi r8 r12 r13 r14 r15 xmm1 xmm2
..B8.42:                        # Preds ..B8.41 ..B8.125
                                # Execution count [8.40e-01]
  0221a 49 89 fa         movq %rdi, %r10                        #../src/feautrier.cpp:264.5
  0221d 49 f7 da         negq %r10                              #../src/feautrier.cpp:264.5
  02220 48 3b f9         cmpq %rcx, %rdi                        #../src/feautrier.cpp:264.5
  02223 73 52            jae ..B8.46 # Prob 9%                  #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp rsi rdi r8 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.43:                        # Preds ..B8.42
                                # Execution count [7.56e-01]
  02225 4c 8d 1c f5 00 
        00 00 00         lea (,%rsi,8), %r11                    #../src/feautrier.cpp:266.26
  0222d 49 2b d3         subq %r11, %rdx                        #../src/feautrier.cpp:266.57
  02230 4d 8d 0c f7      lea (%r15,%rsi,8), %r9                 #../src/feautrier.cpp:266.7
  02234 49 8d 04 f6      lea (%r14,%rsi,8), %rax                #../src/feautrier.cpp:267.7
                                # LOE rax rdx rcx rbx rbp rdi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.44:                        # Preds ..B8.44 ..B8.43
                                # Execution count [4.20e+00]
  02238 f2 42 0f 10 5c 
        d2 f0            movsd -16(%rdx,%r10,8), %xmm3          #../src/feautrier.cpp:266.26
  0223f 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:266.41
  02242 f2 42 0f 10 6c 
        d2 e8            movsd -24(%rdx,%r10,8), %xmm5          #../src/feautrier.cpp:266.41
  02249 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:266.41
  0224c 49 ff ca         decq %r10                              #../src/feautrier.cpp:264.5
  0224f f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:266.41
  02253 f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:266.41
  02257 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:266.57
  0225a f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:266.57
  0225e f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:267.57
  02262 f2 41 0f 11 64 
        f9 08            movsd %xmm4, 8(%r9,%rdi,8)             #../src/feautrier.cpp:266.7
  02269 f2 0f 11 74 f8 
        08               movsd %xmm6, 8(%rax,%rdi,8)            #../src/feautrier.cpp:267.7
  0226f 48 ff c7         incq %rdi                              #../src/feautrier.cpp:264.5
  02272 48 3b f9         cmpq %rcx, %rdi                        #../src/feautrier.cpp:264.5
  02275 72 c1            jb ..B8.44 # Prob 82%                  #../src/feautrier.cpp:264.5
                                # LOE rax rdx rcx rbx rbp rdi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.46:                        # Preds ..B8.20 ..B8.44 ..B8.124 ..B8.42
                                # Execution count [4.20e-01]
  02277 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:272.12
  0227a 7f 0e            jg ..B8.48 # Prob 84%                  #../src/feautrier.cpp:272.12
  0227c e9 df 06 00 00   jmp ..B8.109 # Prob 100%               #../src/feautrier.cpp:272.12
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.47:                        # Preds ..B8.37
                                # Execution count [3.78e-01]
  02281 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:272.12
  02284 0f 8e 80 07 00 
        00               jle ..B8.123 # Prob 16%                #../src/feautrier.cpp:272.12
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.48:                        # Preds ..B8.47 ..B8.46 ..B8.127
                                # Execution count [8.40e-01]
  0228a 49 83 fd 01      cmpq $1, %r13                          #../src/feautrier.cpp:274.30
  0228e 0f 8e 2c 03 00 
        00               jle ..B8.74 # Prob 50%                 #../src/feautrier.cpp:274.30
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.49:                        # Preds ..B8.48
                                # Execution count [7.56e-01]
  02294 4d 8d 4d ff      lea -1(%r13), %r9                      #../src/feautrier.cpp:274.5
  02298 49 83 f9 08      cmpq $8, %r9                           #../src/feautrier.cpp:274.5
  0229c 0f 8c 54 07 00 
        00               jl ..B8.121 # Prob 10%                 #../src/feautrier.cpp:274.5
                                # LOE rbx rbp r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.50:                        # Preds ..B8.49
                                # Execution count [7.56e-01]
  022a2 48 8b 44 24 48   movq 72(%rsp), %rax                    #../src/feautrier.cpp:274.5[spill]
  022a7 48 83 e0 0f      andq $15, %rax                         #../src/feautrier.cpp:274.5
  022ab 74 11            je ..B8.53 # Prob 50%                  #../src/feautrier.cpp:274.5
                                # LOE rax rbx rbp r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.51:                        # Preds ..B8.50
                                # Execution count [7.56e-01]
  022ad 48 a9 07 00 00 
        00               testq $7, %rax                         #../src/feautrier.cpp:274.5
  022b3 0f 85 3d 07 00 
        00               jne ..B8.121 # Prob 10%                #../src/feautrier.cpp:274.5
                                # LOE rbx rbp r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.52:                        # Preds ..B8.51
                                # Execution count [3.78e-01]
  022b9 b8 01 00 00 00   movl $1, %eax                          #../src/feautrier.cpp:274.5
                                # LOE rax rbx rbp r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.53:                        # Preds ..B8.52 ..B8.50
                                # Execution count [7.56e-01]
  022be 48 8d 50 08      lea 8(%rax), %rdx                      #../src/feautrier.cpp:274.5
  022c2 4c 3b ca         cmpq %rdx, %r9                         #../src/feautrier.cpp:274.5
  022c5 0f 8c 2b 07 00 
        00               jl ..B8.121 # Prob 10%                 #../src/feautrier.cpp:274.5
                                # LOE rax rbx rbp r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.54:                        # Preds ..B8.53
                                # Execution count [8.40e-01]
  022cb 4d 89 ca         movq %r9, %r10                         #../src/feautrier.cpp:274.5
  022ce 33 d2            xorl %edx, %edx                        #../src/feautrier.cpp:274.5
  022d0 4c 2b d0         subq %rax, %r10                        #../src/feautrier.cpp:274.5
  022d3 4b 8d 0c e7      lea (%r15,%r12,8), %rcx                #../src/feautrier.cpp:276.7
  022d7 49 83 e2 07      andq $7, %r10                          #../src/feautrier.cpp:274.5
  022db 4f 8d 1c e6      lea (%r14,%r12,8), %r11                #../src/feautrier.cpp:277.7
  022df 49 f7 da         negq %r10                              #../src/feautrier.cpp:274.5
  022e2 4d 03 d1         addq %r9, %r10                         #../src/feautrier.cpp:274.5
  022e5 48 85 c0         testq %rax, %rax                       #../src/feautrier.cpp:274.5
  022e8 76 3c            jbe ..B8.58 # Prob 9%                  #../src/feautrier.cpp:274.5
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B8.55:                        # Preds ..B8.54
                                # Execution count [7.56e-01]
  022ea 48 8b 74 24 48   movq 72(%rsp), %rsi                    #[spill]
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B8.56:                        # Preds ..B8.55 ..B8.56
                                # Execution count [4.20e+00]
  022ef f2 0f 10 1c d6   movsd (%rsi,%rdx,8), %xmm3             #../src/feautrier.cpp:276.26
  022f4 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:276.38
  022f7 f2 0f 10 6c d6 
        08               movsd 8(%rsi,%rdx,8), %xmm5            #../src/feautrier.cpp:276.38
  022fd 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:276.38
  02300 f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:276.38
  02304 f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:276.38
  02308 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:276.53
  0230b f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:276.53
  0230f f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:277.53
  02313 f2 0f 11 24 d1   movsd %xmm4, (%rcx,%rdx,8)             #../src/feautrier.cpp:276.7
  02318 f2 41 0f 11 34 
        d3               movsd %xmm6, (%r11,%rdx,8)             #../src/feautrier.cpp:277.7
  0231e 48 ff c2         incq %rdx                              #../src/feautrier.cpp:274.5
  02321 48 3b d0         cmpq %rax, %rdx                        #../src/feautrier.cpp:274.5
  02324 72 c9            jb ..B8.56 # Prob 82%                  #../src/feautrier.cpp:274.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm1 xmm2
..B8.58:                        # Preds ..B8.56 ..B8.54
                                # Execution count [0.00e+00]
  02326 0f 10 05 fc ff 
        ff ff            movups .L_2il0floatpacket.61(%rip), %xmm0 #../src/feautrier.cpp:276.19
  0232d 49 8d 14 04      lea (%r12,%rax), %rdx                  #../src/feautrier.cpp:277.14
  02331 49 8d 34 d6      lea (%r14,%rdx,8), %rsi                #../src/feautrier.cpp:277.7
  02335 48 f7 c6 0f 00 
        00 00            testq $15, %rsi                        #../src/feautrier.cpp:274.5
  0233c 0f 84 e3 00 00 
        00               je ..B8.62 # Prob 60%                  #../src/feautrier.cpp:274.5
                                # LOE rax rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.59:                        # Preds ..B8.58
                                # Execution count [7.56e-01]
  02342 48 8b 54 24 48   movq 72(%rsp), %rdx                    #[spill]
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.60:                        # Preds ..B8.59 ..B8.60
                                # Execution count [4.20e+00]
  02347 0f 10 24 c2      movups (%rdx,%rax,8), %xmm4            #../src/feautrier.cpp:276.26
  0234b 0f 10 74 c2 08   movups 8(%rdx,%rax,8), %xmm6           #../src/feautrier.cpp:276.26
  02350 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:276.38
  02353 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:276.38
  02356 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:276.38
  0235a 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:276.38
  0235e 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:276.53
  02361 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:276.38
  02365 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:276.53
  02369 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:277.53
  0236d 0f 11 2c c1      movups %xmm5, (%rcx,%rax,8)            #../src/feautrier.cpp:276.7
  02371 41 0f 11 3c c3   movups %xmm7, (%r11,%rax,8)            #../src/feautrier.cpp:277.7
  02376 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:276.38
  02379 44 0f 10 4c c2 
        10               movups 16(%rdx,%rax,8), %xmm9          #../src/feautrier.cpp:276.26
  0237f 44 0f 10 5c c2 
        18               movups 24(%rdx,%rax,8), %xmm11         #../src/feautrier.cpp:276.26
  02385 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:276.38
  02389 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:276.38
  0238e 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:276.38
  02393 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:276.53
  02397 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:276.53
  0239c 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:277.53
  023a1 44 0f 11 54 c1 
        10               movups %xmm10, 16(%rcx,%rax,8)         #../src/feautrier.cpp:276.7
  023a7 45 0f 11 64 c3 
        10               movups %xmm12, 16(%r11,%rax,8)         #../src/feautrier.cpp:277.7
  023ad 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:276.38
  023b1 44 0f 10 74 c2 
        20               movups 32(%rdx,%rax,8), %xmm14         #../src/feautrier.cpp:276.26
  023b7 0f 10 5c c2 28   movups 40(%rdx,%rax,8), %xmm3          #../src/feautrier.cpp:276.26
  023bc 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:276.38
  023c0 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:276.38
  023c5 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:276.38
  023ca 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:276.53
  023ce 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:276.53
  023d3 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:277.53
  023d7 44 0f 11 7c c1 
        20               movups %xmm15, 32(%rcx,%rax,8)         #../src/feautrier.cpp:276.7
  023dd 41 0f 11 64 c3 
        20               movups %xmm4, 32(%r11,%rax,8)          #../src/feautrier.cpp:277.7
  023e3 0f 10 74 c2 30   movups 48(%rdx,%rax,8), %xmm6          #../src/feautrier.cpp:276.26
  023e8 44 0f 10 44 c2 
        38               movups 56(%rdx,%rax,8), %xmm8          #../src/feautrier.cpp:276.26
  023ee 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:276.38
  023f1 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:276.38
  023f6 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:276.38
  023fb 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:276.53
  023ff 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:276.53
  02403 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:277.53
  02408 0f 11 7c c1 30   movups %xmm7, 48(%rcx,%rax,8)          #../src/feautrier.cpp:276.7
  0240d 45 0f 11 4c c3 
        30               movups %xmm9, 48(%r11,%rax,8)          #../src/feautrier.cpp:277.7
  02413 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:274.5
  02417 49 3b c2         cmpq %r10, %rax                        #../src/feautrier.cpp:274.5
  0241a 0f 82 27 ff ff 
        ff               jb ..B8.60 # Prob 82%                  #../src/feautrier.cpp:274.5
  02420 e9 de 00 00 00   jmp ..B8.65 # Prob 100%                #../src/feautrier.cpp:274.5
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.62:                        # Preds ..B8.58
                                # Execution count [7.56e-01]
  02425 48 8b 54 24 48   movq 72(%rsp), %rdx                    #[spill]
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.63:                        # Preds ..B8.62 ..B8.63
                                # Execution count [4.20e+00]
  0242a 0f 10 24 c2      movups (%rdx,%rax,8), %xmm4            #../src/feautrier.cpp:276.26
  0242e 0f 10 74 c2 08   movups 8(%rdx,%rax,8), %xmm6           #../src/feautrier.cpp:276.26
  02433 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:276.38
  02436 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:276.38
  02439 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:276.38
  0243d 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:276.38
  02441 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:276.53
  02444 44 0f 28 e0      movaps %xmm0, %xmm12                   #../src/feautrier.cpp:276.38
  02448 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:276.53
  0244c 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:277.53
  02450 0f 11 2c c1      movups %xmm5, (%rcx,%rax,8)            #../src/feautrier.cpp:276.7
  02454 41 0f 11 3c c3   movups %xmm7, (%r11,%rax,8)            #../src/feautrier.cpp:277.7
  02459 0f 28 e0         movaps %xmm0, %xmm4                    #../src/feautrier.cpp:276.38
  0245c 44 0f 10 4c c2 
        10               movups 16(%rdx,%rax,8), %xmm9          #../src/feautrier.cpp:276.26
  02462 44 0f 10 5c c2 
        18               movups 24(%rdx,%rax,8), %xmm11         #../src/feautrier.cpp:276.26
  02468 45 0f 28 c1      movaps %xmm9, %xmm8                    #../src/feautrier.cpp:276.38
  0246c 66 45 0f 58 c3   addpd %xmm11, %xmm8                    #../src/feautrier.cpp:276.38
  02471 66 45 0f 5e e0   divpd %xmm8, %xmm12                    #../src/feautrier.cpp:276.38
  02476 45 0f 28 d4      movaps %xmm12, %xmm10                  #../src/feautrier.cpp:276.53
  0247a 66 45 0f 5e d1   divpd %xmm9, %xmm10                    #../src/feautrier.cpp:276.53
  0247f 66 45 0f 5e e3   divpd %xmm11, %xmm12                   #../src/feautrier.cpp:277.53
  02484 44 0f 11 54 c1 
        10               movups %xmm10, 16(%rcx,%rax,8)         #../src/feautrier.cpp:276.7
  0248a 45 0f 11 64 c3 
        10               movups %xmm12, 16(%r11,%rax,8)         #../src/feautrier.cpp:277.7
  02490 44 0f 28 c8      movaps %xmm0, %xmm9                    #../src/feautrier.cpp:276.38
  02494 44 0f 10 74 c2 
        20               movups 32(%rdx,%rax,8), %xmm14         #../src/feautrier.cpp:276.26
  0249a 0f 10 5c c2 28   movups 40(%rdx,%rax,8), %xmm3          #../src/feautrier.cpp:276.26
  0249f 45 0f 28 ee      movaps %xmm14, %xmm13                  #../src/feautrier.cpp:276.38
  024a3 66 44 0f 58 eb   addpd %xmm3, %xmm13                    #../src/feautrier.cpp:276.38
  024a8 66 41 0f 5e e5   divpd %xmm13, %xmm4                    #../src/feautrier.cpp:276.38
  024ad 44 0f 28 fc      movaps %xmm4, %xmm15                   #../src/feautrier.cpp:276.53
  024b1 66 45 0f 5e fe   divpd %xmm14, %xmm15                   #../src/feautrier.cpp:276.53
  024b6 66 0f 5e e3      divpd %xmm3, %xmm4                     #../src/feautrier.cpp:277.53
  024ba 44 0f 11 7c c1 
        20               movups %xmm15, 32(%rcx,%rax,8)         #../src/feautrier.cpp:276.7
  024c0 41 0f 11 64 c3 
        20               movups %xmm4, 32(%r11,%rax,8)          #../src/feautrier.cpp:277.7
  024c6 0f 10 74 c2 30   movups 48(%rdx,%rax,8), %xmm6          #../src/feautrier.cpp:276.26
  024cb 44 0f 10 44 c2 
        38               movups 56(%rdx,%rax,8), %xmm8          #../src/feautrier.cpp:276.26
  024d1 0f 28 ee         movaps %xmm6, %xmm5                    #../src/feautrier.cpp:276.38
  024d4 66 41 0f 58 e8   addpd %xmm8, %xmm5                     #../src/feautrier.cpp:276.38
  024d9 66 44 0f 5e cd   divpd %xmm5, %xmm9                     #../src/feautrier.cpp:276.38
  024de 41 0f 28 f9      movaps %xmm9, %xmm7                    #../src/feautrier.cpp:276.53
  024e2 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:276.53
  024e6 66 45 0f 5e c8   divpd %xmm8, %xmm9                     #../src/feautrier.cpp:277.53
  024eb 0f 11 7c c1 30   movups %xmm7, 48(%rcx,%rax,8)          #../src/feautrier.cpp:276.7
  024f0 45 0f 11 4c c3 
        30               movups %xmm9, 48(%r11,%rax,8)          #../src/feautrier.cpp:277.7
  024f6 48 83 c0 08      addq $8, %rax                          #../src/feautrier.cpp:274.5
  024fa 49 3b c2         cmpq %r10, %rax                        #../src/feautrier.cpp:274.5
  024fd 0f 82 27 ff ff 
        ff               jb ..B8.63 # Prob 82%                  #../src/feautrier.cpp:274.5
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.65:                        # Preds ..B8.63 ..B8.60
                                # Execution count [7.56e-01]
  02503 49 8d 42 01      lea 1(%r10), %rax                      #../src/feautrier.cpp:274.5
  02507 49 3b c1         cmpq %r9, %rax                         #../src/feautrier.cpp:274.5
  0250a 0f 87 b0 00 00 
        00               ja ..B8.74 # Prob 50%                  #../src/feautrier.cpp:274.5
                                # LOE rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.66:                        # Preds ..B8.65
                                # Execution count [7.56e-01]
  02510 4d 2b ca         subq %r10, %r9                         #../src/feautrier.cpp:274.5
  02513 49 83 f9 02      cmpq $2, %r9                           #../src/feautrier.cpp:274.5
  02517 0f 8c e6 04 00 
        00               jl ..B8.122 # Prob 10%                 #../src/feautrier.cpp:274.5
                                # LOE rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.67:                        # Preds ..B8.66
                                # Execution count [7.56e-01]
  0251d 4c 89 ce         movq %r9, %rsi                         #../src/feautrier.cpp:274.5
  02520 33 d2            xorl %edx, %edx                        #../src/feautrier.cpp:274.5
  02522 48 8b 7c 24 48   movq 72(%rsp), %rdi                    #../src/feautrier.cpp:277.7[spill]
  02527 48 83 e6 fe      andq $-2, %rsi                         #../src/feautrier.cpp:274.5
  0252b 4a 8d 04 d1      lea (%rcx,%r10,8), %rax                #../src/feautrier.cpp:276.7
  0252f 4b 8d 0c d3      lea (%r11,%r10,8), %rcx                #../src/feautrier.cpp:277.7
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.68:                        # Preds ..B8.68 ..B8.67
                                # Execution count [4.20e+00]
  02533 0f 28 f8         movaps %xmm0, %xmm7                    #../src/feautrier.cpp:276.38
  02536 4e 8d 1c 12      lea (%rdx,%r10), %r11                  #../src/feautrier.cpp:276.26
  0253a 42 0f 10 24 df   movups (%rdi,%r11,8), %xmm4            #../src/feautrier.cpp:276.26
  0253f 42 0f 10 74 df 
        08               movups 8(%rdi,%r11,8), %xmm6           #../src/feautrier.cpp:276.26
  02545 0f 28 dc         movaps %xmm4, %xmm3                    #../src/feautrier.cpp:276.38
  02548 66 0f 58 de      addpd %xmm6, %xmm3                     #../src/feautrier.cpp:276.38
  0254c 66 0f 5e fb      divpd %xmm3, %xmm7                     #../src/feautrier.cpp:276.38
  02550 0f 28 ef         movaps %xmm7, %xmm5                    #../src/feautrier.cpp:276.53
  02553 66 0f 5e ec      divpd %xmm4, %xmm5                     #../src/feautrier.cpp:276.53
  02557 66 0f 5e fe      divpd %xmm6, %xmm7                     #../src/feautrier.cpp:277.53
  0255b 0f 11 2c d0      movups %xmm5, (%rax,%rdx,8)            #../src/feautrier.cpp:276.7
  0255f 0f 11 3c d1      movups %xmm7, (%rcx,%rdx,8)            #../src/feautrier.cpp:277.7
  02563 48 83 c2 02      addq $2, %rdx                          #../src/feautrier.cpp:274.5
  02567 48 3b d6         cmpq %rsi, %rdx                        #../src/feautrier.cpp:274.5
  0256a 72 c7            jb ..B8.68 # Prob 82%                  #../src/feautrier.cpp:274.5
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r12 r13 r14 r15 xmm0 xmm1 xmm2
..B8.70:                        # Preds ..B8.68 ..B8.122
                                # Execution count [8.40e-01]
  0256c 49 3b f1         cmpq %r9, %rsi                         #../src/feautrier.cpp:274.5
  0256f 73 4f            jae ..B8.74 # Prob 9%                  #../src/feautrier.cpp:274.5
                                # LOE rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.71:                        # Preds ..B8.70
                                # Execution count [7.56e-01]
  02571 48 8b 44 24 48   movq 72(%rsp), %rax                    #../src/feautrier.cpp:276.53[spill]
  02576 4b 8d 0c e7      lea (%r15,%r12,8), %rcx                #../src/feautrier.cpp:276.7
  0257a 4b 8d 14 e6      lea (%r14,%r12,8), %rdx                #../src/feautrier.cpp:277.7
  0257e 4a 8d 0c d1      lea (%rcx,%r10,8), %rcx                #../src/feautrier.cpp:276.7
  02582 4a 8d 14 d2      lea (%rdx,%r10,8), %rdx                #../src/feautrier.cpp:277.7
  02586 4a 8d 04 d0      lea (%rax,%r10,8), %rax                #../src/feautrier.cpp:276.53
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.72:                        # Preds ..B8.72 ..B8.71
                                # Execution count [4.20e+00]
  0258a f2 0f 10 1c f0   movsd (%rax,%rsi,8), %xmm3             #../src/feautrier.cpp:276.26
  0258f 0f 28 f1         movaps %xmm1, %xmm6                    #../src/feautrier.cpp:276.38
  02592 f2 0f 10 6c f0 
        08               movsd 8(%rax,%rsi,8), %xmm5            #../src/feautrier.cpp:276.38
  02598 0f 28 c3         movaps %xmm3, %xmm0                    #../src/feautrier.cpp:276.38
  0259b f2 0f 58 c5      addsd %xmm5, %xmm0                     #../src/feautrier.cpp:276.38
  0259f f2 0f 5e f0      divsd %xmm0, %xmm6                     #../src/feautrier.cpp:276.38
  025a3 0f 28 e6         movaps %xmm6, %xmm4                    #../src/feautrier.cpp:276.53
  025a6 f2 0f 5e e3      divsd %xmm3, %xmm4                     #../src/feautrier.cpp:276.53
  025aa f2 0f 5e f5      divsd %xmm5, %xmm6                     #../src/feautrier.cpp:277.53
  025ae f2 0f 11 24 f1   movsd %xmm4, (%rcx,%rsi,8)             #../src/feautrier.cpp:276.7
  025b3 f2 0f 11 34 f2   movsd %xmm6, (%rdx,%rsi,8)             #../src/feautrier.cpp:277.7
  025b8 48 ff c6         incq %rsi                              #../src/feautrier.cpp:274.5
  025bb 49 3b f1         cmpq %r9, %rsi                         #../src/feautrier.cpp:274.5
  025be 72 ca            jb ..B8.72 # Prob 82%                  #../src/feautrier.cpp:274.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r12 r13 r14 r15 xmm1 xmm2
..B8.74:                        # Preds ..B8.72 ..B8.121 ..B8.70 ..B8.48 ..B8.65
                                #      
                                # Execution count [8.40e-01]
  025c0 48 8b 54 24 48   movq 72(%rsp), %rdx                    #../src/feautrier.cpp:280.21[spill]
  025c5 0f 28 e1         movaps %xmm1, %xmm4                    #../src/feautrier.cpp:280.21
  025c8 49 8d 0c df      lea (%r15,%rbx,8), %rcx                #../src/feautrier.cpp:280.5
  025cc 66 0f ef db      pxor %xmm3, %xmm3                      #../src/feautrier.cpp:281.5
  025d0 48 8d 04 dd 00 
        00 00 00         lea (,%rbx,8), %rax                    #../src/feautrier.cpp:280.5
  025d8 f2 42 0f 10 44 
        ea f8            movsd -8(%rdx,%r13,8), %xmm0           #../src/feautrier.cpp:280.21
  025df f2 0f 5e e0      divsd %xmm0, %xmm4                     #../src/feautrier.cpp:280.21
  025e3 0f 28 ec         movaps %xmm4, %xmm5                    #../src/feautrier.cpp:280.35
  025e6 f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:283.29
  025ea f2 0f 5e e8      divsd %xmm0, %xmm5                     #../src/feautrier.cpp:280.35
  025ee f2 0f 11 69 f8   movsd %xmm5, -8(%rcx)                  #../src/feautrier.cpp:280.5
  025f3 f2 0f 58 ec      addsd %xmm4, %xmm5                     #../src/feautrier.cpp:283.63
  025f7 f2 41 0f 11 5c 
        06 f8            movsd %xmm3, -8(%r14,%rax)             #../src/feautrier.cpp:281.5
  025fe f2 0f 11 64 24 
        38               movsd %xmm4, 56(%rsp)                  #../src/feautrier.cpp:283.29[spill]
  02604 f2 0f 11 6c 24 
        28               movsd %xmm5, 40(%rsp)                  #../src/feautrier.cpp:283.63[spill]
  0260a 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:288.14
  0260d 75 39            jne ..B8.76 # Prob 50%                 #../src/feautrier.cpp:288.14
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.75:                        # Preds ..B8.74 ..B8.129
                                # Execution count [5.00e-01]
  0260f 48 8b 54 24 48   movq 72(%rsp), %rdx                    #../src/feautrier.cpp:291.16[spill]
  02614 0f 28 e2         movaps %xmm2, %xmm4                    #../src/feautrier.cpp:293.27
  02617 66 0f ef c0      pxor %xmm0, %xmm0                      #../src/feautrier.cpp:290.5
  0261b f2 0f 10 1a      movsd (%rdx), %xmm3                    #../src/feautrier.cpp:291.16
  0261f f2 0f 5e cb      divsd %xmm3, %xmm1                     #../src/feautrier.cpp:291.16
  02623 0f 28 e9         movaps %xmm1, %xmm5                    #../src/feautrier.cpp:291.26
  02626 f2 0f 58 e1      addsd %xmm1, %xmm4                     #../src/feautrier.cpp:293.27
  0262a f2 0f 5e eb      divsd %xmm3, %xmm5                     #../src/feautrier.cpp:291.26
  0262e f2 41 0f 11 2e   movsd %xmm5, (%r14)                    #../src/feautrier.cpp:291.5
  02633 f2 0f 58 ec      addsd %xmm4, %xmm5                     #../src/feautrier.cpp:293.53
  02637 f2 41 0f 11 07   movsd %xmm0, (%r15)                    #../src/feautrier.cpp:290.5
  0263c f2 0f 11 64 24 
        40               movsd %xmm4, 64(%rsp)                  #../src/feautrier.cpp:293.27[spill]
  02642 f2 0f 11 6c 24 
        30               movsd %xmm5, 48(%rsp)                  #../src/feautrier.cpp:293.53[spill]
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.76:                        # Preds ..B8.74 ..B8.75 ..B8.130
                                # Execution count [8.72e-01]
  02648 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:298.13
  0264b 75 50            jne ..B8.79 # Prob 50%                 #../src/feautrier.cpp:298.13
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.77:                        # Preds ..B8.76
                                # Execution count [4.36e-01]
  0264d f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:300.17
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.78:                        # Preds ..B8.109 ..B8.123 ..B8.77
                                # Execution count [5.00e-01]
  02655 48 8b 54 24 58   movq 88(%rsp), %rdx                    #../src/feautrier.cpp:300.21[spill]
  0265a 0f 28 ea         movaps %xmm2, %xmm5                    #../src/feautrier.cpp:303.29
  0265d 48 83 7c 24 18 
        00               cmpq $0, 24(%rsp)                      #../src/feautrier.cpp:310.30[spill]
  02663 66 0f ef e4      pxor %xmm4, %xmm4                      #../src/feautrier.cpp:301.5
  02667 f2 0f 10 02      movsd (%rdx), %xmm0                    #../src/feautrier.cpp:300.21
  0266b f2 0f 5e c8      divsd %xmm0, %xmm1                     #../src/feautrier.cpp:300.21
  0266f 0f 28 d9         movaps %xmm1, %xmm3                    #../src/feautrier.cpp:300.32
  02672 f2 0f 58 e9      addsd %xmm1, %xmm5                     #../src/feautrier.cpp:303.29
  02676 f2 0f 5e d8      divsd %xmm0, %xmm3                     #../src/feautrier.cpp:300.32
  0267a f2 0f 11 6c 24 
        38               movsd %xmm5, 56(%rsp)                  #../src/feautrier.cpp:303.29[spill]
  02680 f2 0f 58 eb      addsd %xmm3, %xmm5                     #../src/feautrier.cpp:303.57
  02684 f2 0f 11 59 f8   movsd %xmm3, -8(%rcx)                  #../src/feautrier.cpp:300.5
  02689 f2 42 0f 11 64 
        30 f8            movsd %xmm4, -8(%rax,%r14)             #../src/feautrier.cpp:301.5
  02690 f2 0f 11 6c 24 
        28               movsd %xmm5, 40(%rsp)                  #../src/feautrier.cpp:303.57[spill]
  02696 7d 0d            jge ..B8.80 # Prob 50%                 #../src/feautrier.cpp:310.30
  02698 e9 f9 00 00 00   jmp ..B8.92 # Prob 100%                #../src/feautrier.cpp:310.30
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2 xmm3
..B8.79:                        # Preds ..B8.76
                                # Execution count [4.36e-01]
  0269d 48 83 7c 24 18 
        00               cmpq $0, 24(%rsp)                      #../src/feautrier.cpp:310.30[spill]
  026a3 7c 70            jl ..B8.86 # Prob 50%                  #../src/feautrier.cpp:310.30
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.80:                        # Preds ..B8.123 ..B8.110 ..B8.78 ..B8.79
                                # Execution count [1.00e+00]
  026a5 4c 89 e6         movq %r12, %rsi                        #../src/feautrier.cpp:310.3
  026a8 45 33 c9         xorl %r9d, %r9d                        #../src/feautrier.cpp:310.3
  026ab 48 c1 ee 3f      shrq $63, %rsi                         #../src/feautrier.cpp:310.3
  026af 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:312.5
  026b2 49 03 f4         addq %r12, %rsi                        #../src/feautrier.cpp:226.5
  026b5 ba 01 00 00 00   movl $1, %edx                          #../src/feautrier.cpp:310.3
  026ba 48 d1 fe         sarq $1, %rsi                          #../src/feautrier.cpp:226.5
  026bd 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:312.5
  026c0 48 85 f6         testq %rsi, %rsi                       #../src/feautrier.cpp:310.3
  026c3 76 34            jbe ..B8.84 # Prob 2%                  #../src/feautrier.cpp:310.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B8.81:                        # Preds ..B8.80
                                # Execution count [9.79e-01]
  026c5 48 8b 54 24 50   movq 80(%rsp), %rdx                    #../src/feautrier.cpp:312.19[spill]
  026ca 4a 8d 14 e2      lea (%rdx,%r12,8), %rdx                #../src/feautrier.cpp:312.19
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B8.82:                        # Preds ..B8.82 ..B8.81
                                # Execution count [2.72e+00]
  026ce 49 8b 7c 12 f8   movq -8(%r10,%rdx), %rdi               #../src/feautrier.cpp:312.19
  026d3 49 ff c1         incq %r9                               #../src/feautrier.cpp:310.3
  026d6 4b 89 3c 03      movq %rdi, (%r11,%r8)                  #../src/feautrier.cpp:312.5
  026da 49 8b 7c 12 f0   movq -16(%r10,%rdx), %rdi              #../src/feautrier.cpp:312.19
  026df 49 83 c2 f0      addq $-16, %r10                        #../src/feautrier.cpp:310.3
  026e3 4b 89 7c 03 08   movq %rdi, 8(%r11,%r8)                 #../src/feautrier.cpp:312.5
  026e8 49 83 c3 10      addq $16, %r11                         #../src/feautrier.cpp:310.3
  026ec 4c 3b ce         cmpq %rsi, %r9                         #../src/feautrier.cpp:310.3
  026ef 72 dd            jb ..B8.82 # Prob 63%                  #../src/feautrier.cpp:310.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B8.83:                        # Preds ..B8.82
                                # Execution count [9.79e-01]
  026f1 4a 8d 14 4d 01 
        00 00 00         lea 1(,%r9,2), %rdx                    #../src/feautrier.cpp:312.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.84:                        # Preds ..B8.83 ..B8.80
                                # Execution count [1.00e+00]
  026f9 48 ff ca         decq %rdx                              #../src/feautrier.cpp:312.5
  026fc 49 3b d4         cmpq %r12, %rdx                        #../src/feautrier.cpp:310.3
  026ff 73 14            jae ..B8.86 # Prob 2%                  #../src/feautrier.cpp:310.3
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.85:                        # Preds ..B8.84
                                # Execution count [9.79e-01]
  02701 4d 89 e1         movq %r12, %r9                         #../src/feautrier.cpp:312.5
  02704 4c 2b ca         subq %rdx, %r9                         #../src/feautrier.cpp:312.5
  02707 48 8b 74 24 50   movq 80(%rsp), %rsi                    #../src/feautrier.cpp:312.19[spill]
  0270c 4e 8b 54 ce f8   movq -8(%rsi,%r9,8), %r10              #../src/feautrier.cpp:312.19
  02711 4d 89 14 d0      movq %r10, (%r8,%rdx,8)                #../src/feautrier.cpp:312.5
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.86:                        # Preds ..B8.85 ..B8.84 ..B8.79
                                # Execution count [7.18e-01]
  02715 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:315.24
  02718 7e 77            jle ..B8.91 # Prob 50%                 #../src/feautrier.cpp:315.24
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.87:                        # Preds ..B8.86
                                # Execution count [5.00e-03]
  0271a 49 83 fd 0c      cmpq $12, %r13                         #../src/feautrier.cpp:315.3
  0271e 0f 8e 5f 02 00 
        00               jle ..B8.113 # Prob 10%                #../src/feautrier.cpp:315.3
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.88:                        # Preds ..B8.87
                                # Execution count [1.00e+00]
  02724 41 b9 01 00 00 
        00               movl $1, %r9d                          #../src/feautrier.cpp:315.3
  0272a 4b 8d 3c e0      lea (%r8,%r12,8), %rdi                 #../src/feautrier.cpp:317.5
  0272e 48 89 fe         movq %rdi, %rsi                        #../src/feautrier.cpp:315.3
  02731 4a 8d 14 ed 00 
        00 00 00         lea (,%r13,8), %rdx                    #../src/feautrier.cpp:315.3
  02739 48 2b 74 24 20   subq 32(%rsp), %rsi                    #../src/feautrier.cpp:315.3[spill]
  0273e 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:315.3
  02741 48 3b f2         cmpq %rdx, %rsi                        #../src/feautrier.cpp:315.3
  02744 45 0f 4f d9      cmovg %r9d, %r11d                      #../src/feautrier.cpp:315.3
  02748 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:315.3
  0274b 48 f7 de         negq %rsi                              #../src/feautrier.cpp:315.3
  0274e 48 3b f2         cmpq %rdx, %rsi                        #../src/feautrier.cpp:315.3
  02751 45 0f 4f d1      cmovg %r9d, %r10d                      #../src/feautrier.cpp:315.3
  02755 45 0b da         orl %r10d, %r11d                       #../src/feautrier.cpp:315.3
  02758 0f 84 25 02 00 
        00               je ..B8.113 # Prob 10%                 #../src/feautrier.cpp:315.3
                                # LOE rax rdx rcx rbx rbp rdi r8 r12 r13 r14 r15 xmm2
..B8.89:                        # Preds ..B8.88
                                # Execution count [1.00e+00]
  0275e 48 8b 74 24 20   movq 32(%rsp), %rsi                    #../src/feautrier.cpp:317.5[spill]
  02763 48 89 0c 24      movq %rcx, (%rsp)                      #../src/feautrier.cpp:317.5[spill]
  02767 48 89 44 24 08   movq %rax, 8(%rsp)                     #../src/feautrier.cpp:317.5[spill]
  0276c e8 fc ff ff ff   call _intel_fast_memcpy                #../src/feautrier.cpp:317.5
                                # LOE rbx rbp r12 r13 r14 r15
..B8.90:                        # Preds ..B8.89
                                # Execution count [1.00e+00]
  02771 48 8b 0c 24      movq (%rsp), %rcx                      #[spill]
  02775 48 8b 44 24 08   movq 8(%rsp), %rax                     #[spill]
  0277a 4c 8b 84 24 a0 
        00 00 00         movq 160(%rsp), %r8                    #
  02782 f2 0f 10 15 fc 
        ff ff ff         movsd .L_2il0floatpacket.62(%rip), %xmm2 #
  0278a f2 0f 10 59 f8   movsd -8(%rcx), %xmm3                  #../src/feautrier.cpp:342.28
  0278f eb 05            jmp ..B8.92 # Prob 100%                #../src/feautrier.cpp:342.28
                                # LOE rax rbx rbp r8 r12 r13 r14 r15 xmm2 xmm3
..B8.91:                        # Preds ..B8.117 ..B8.110 ..B8.86
                                # Execution count [3.59e-01]
  02791 f2 0f 10 59 f8   movsd -8(%rcx), %xmm3                  #../src/feautrier.cpp:342.28
                                # LOE rax rbx rbp r8 r12 r13 r14 r15 xmm2 xmm3
..B8.92:                        # Preds ..B8.90 ..B8.118 ..B8.78 ..B8.91
                                # Execution count [1.00e+00]
  02796 f2 41 0f 10 00   movsd (%r8), %xmm0                     #../src/feautrier.cpp:329.10
  0279b 4b 8d 54 25 fe   lea -2(%r13,%r12), %rdx                #../src/feautrier.cpp:334.29
  027a0 f2 0f 10 4c 24 
        40               movsd 64(%rsp), %xmm1                  #../src/feautrier.cpp:331.22[spill]
  027a6 33 c9            xorl %ecx, %ecx                        #../src/feautrier.cpp:334.3
  027a8 f2 0f 5e 44 24 
        30               divsd 48(%rsp), %xmm0                  #../src/feautrier.cpp:329.17[spill]
  027ae f2 41 0f 5e 0e   divsd (%r14), %xmm1                    #../src/feautrier.cpp:331.22
  027b3 f2 41 0f 11 00   movsd %xmm0, (%r8)                     #../src/feautrier.cpp:329.3
  027b8 f2 0f 11 4c 24 
        40               movsd %xmm1, 64(%rsp)                  #../src/feautrier.cpp:331.22[spill]
  027be f2 0f 11 4d 00   movsd %xmm1, (%rbp)                    #../src/feautrier.cpp:331.3
  027c3 48 83 fb 02      cmpq $2, %rbx                          #../src/feautrier.cpp:334.29
  027c7 7e 6c            jle ..B8.96 # Prob 9%                  #../src/feautrier.cpp:334.29
  027c9 0f 1f 80 00 00 
        00 00            .align    16,0x90
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2 xmm3
..B8.94:                        # Preds ..B8.92 ..B8.94
                                # Execution count [5.00e+00]
  027d0 f2 41 0f 10 6c 
        cf 08            movsd 8(%r15,%rcx,8), %xmm5            #../src/feautrier.cpp:336.19
  027d7 f2 0f 10 4c cd 
        00               movsd (%rbp,%rcx,8), %xmm1             #../src/feautrier.cpp:336.24
  027dd 0f 28 e5         movaps %xmm5, %xmm4                    #../src/feautrier.cpp:336.24
  027e0 f2 0f 59 e1      mulsd %xmm1, %xmm4                     #../src/feautrier.cpp:336.24
  027e4 f2 0f 58 ca      addsd %xmm2, %xmm1                     #../src/feautrier.cpp:336.38
  027e8 f2 0f 59 e8      mulsd %xmm0, %xmm5                     #../src/feautrier.cpp:338.25
  027ec f2 0f 5e e1      divsd %xmm1, %xmm4                     #../src/feautrier.cpp:336.38
  027f0 f2 41 0f 10 74 
        ce 08            movsd 8(%r14,%rcx,8), %xmm6            #../src/feautrier.cpp:336.49
  027f7 f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:336.38
  027fb f2 41 0f 58 6c 
        c8 08            addsd 8(%r8,%rcx,8), %xmm5             #../src/feautrier.cpp:338.25
  02802 f2 0f 5e e6      divsd %xmm6, %xmm4                     #../src/feautrier.cpp:336.49
  02806 f2 0f 11 64 cd 
        08               movsd %xmm4, 8(%rbp,%rcx,8)            #../src/feautrier.cpp:336.5
  0280c f2 0f 58 e2      addsd %xmm2, %xmm4                     #../src/feautrier.cpp:338.42
  02810 f2 0f 5e ec      divsd %xmm4, %xmm5                     #../src/feautrier.cpp:338.42
  02814 0f 28 c5         movaps %xmm5, %xmm0                    #../src/feautrier.cpp:338.50
  02817 f2 0f 5e c6      divsd %xmm6, %xmm0                     #../src/feautrier.cpp:338.50
  0281b f2 41 0f 11 44 
        c8 08            movsd %xmm0, 8(%r8,%rcx,8)             #../src/feautrier.cpp:338.5
  02822 48 ff c1         incq %rcx                              #../src/feautrier.cpp:334.3
  02825 48 3b ca         cmpq %rdx, %rcx                        #../src/feautrier.cpp:334.3
  02828 72 a6            jb ..B8.94 # Prob 82%                  #../src/feautrier.cpp:334.3
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2 xmm3
..B8.95:                        # Preds ..B8.94
                                # Execution count [9.00e-01]
  0282a f2 0f 10 45 00   movsd (%rbp), %xmm0                    #../src/feautrier.cpp:353.27
  0282f f2 0f 11 44 24 
        40               movsd %xmm0, 64(%rsp)                  #../src/feautrier.cpp:353.27[spill]
                                # LOE rax rdx rbx rbp r8 r12 r13 r14 r15 xmm2 xmm3
..B8.96:                        # Preds ..B8.92 ..B8.95
                                # Execution count [1.00e+00]
  02835 f2 0f 10 4c 28 
        f0               movsd -16(%rax,%rbp), %xmm1            #../src/feautrier.cpp:343.36
  0283b 0f 28 c2         movaps %xmm2, %xmm0                    #../src/feautrier.cpp:343.56
  0283e f2 42 0f 59 5c 
        00 f0            mulsd -16(%rax,%r8), %xmm3             #../src/feautrier.cpp:342.38
  02845 f2 0f 58 c1      addsd %xmm1, %xmm0                     #../src/feautrier.cpp:343.56
  02849 f2 0f 59 4c 24 
        28               mulsd 40(%rsp), %xmm1                  #../src/feautrier.cpp:343.36[spill]
  0284f f2 42 0f 58 5c 
        00 f8            addsd -8(%rax,%r8), %xmm3              #../src/feautrier.cpp:342.38
  02856 f2 0f 58 4c 24 
        38               addsd 56(%rsp), %xmm1                  #../src/feautrier.cpp:343.36[spill]
  0285c f2 0f 5e d9      divsd %xmm1, %xmm3                     #../src/feautrier.cpp:343.36
  02860 f2 0f 59 c3      mulsd %xmm3, %xmm0                     #../src/feautrier.cpp:343.56
  02864 f2 42 0f 11 44 
        00 f8            movsd %xmm0, -8(%rax,%r8)              #../src/feautrier.cpp:342.3
  0286b 48 85 d2         testq %rdx, %rdx                       #../src/feautrier.cpp:348.29
  0286e 0f 8e 9f 00 00 
        00               jle ..B8.103 # Prob 50%                #../src/feautrier.cpp:348.29
                                # LOE rdx rbx rbp r8 r12 r13 r14 r15 xmm0 xmm2
..B8.97:                        # Preds ..B8.96
                                # Execution count [1.00e+00]
  02874 48 89 d0         movq %rdx, %rax                        #../src/feautrier.cpp:348.3
  02877 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:348.3
  0287a 48 c1 e8 3f      shrq $63, %rax                         #../src/feautrier.cpp:348.3
  0287e b9 01 00 00 00   movl $1, %ecx                          #../src/feautrier.cpp:348.3
  02883 33 f6            xorl %esi, %esi                        #../src/feautrier.cpp:350.5
  02885 4c 8d 4c 03 fe   lea -2(%rbx,%rax), %r9                 #../src/feautrier.cpp:348.3
  0288a 49 d1 f9         sarq $1, %r9                           #../src/feautrier.cpp:348.3
  0288d 4d 85 c9         testq %r9, %r9                         #../src/feautrier.cpp:348.3
  02890 76 54            jbe ..B8.101 # Prob 2%                 #../src/feautrier.cpp:348.3
                                # LOE rdx rcx rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm0 xmm2
..B8.98:                        # Preds ..B8.97
                                # Execution count [9.79e-01]
  02892 4d 03 ec         addq %r12, %r13                        #../src/feautrier.cpp:350.31
  02895 4a 8d 4c ed 00   lea (%rbp,%r13,8), %rcx                #../src/feautrier.cpp:350.31
  0289a 4b 8d 04 e8      lea (%r8,%r13,8), %rax                 #../src/feautrier.cpp:350.12
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r14 r15 xmm0 xmm2
..B8.99:                        # Preds ..B8.99 ..B8.98
                                # Execution count [2.72e+00]
  0289e f2 0f 10 4c 0e 
        f0               movsd -16(%rsi,%rcx), %xmm1            #../src/feautrier.cpp:350.31
  028a4 49 ff c2         incq %r10                              #../src/feautrier.cpp:348.3
  028a7 f2 0f 10 5c 0e 
        e8               movsd -24(%rsi,%rcx), %xmm3            #../src/feautrier.cpp:350.31
  028ad f2 0f 58 ca      addsd %xmm2, %xmm1                     #../src/feautrier.cpp:350.31
  028b1 f2 0f 58 da      addsd %xmm2, %xmm3                     #../src/feautrier.cpp:350.31
  028b5 f2 0f 5e c1      divsd %xmm1, %xmm0                     #../src/feautrier.cpp:350.31
  028b9 f2 0f 58 44 06 
        f0               addsd -16(%rsi,%rax), %xmm0            #../src/feautrier.cpp:350.31
  028bf f2 0f 11 44 06 
        f0               movsd %xmm0, -16(%rsi,%rax)            #../src/feautrier.cpp:350.5
  028c5 f2 0f 5e c3      divsd %xmm3, %xmm0                     #../src/feautrier.cpp:350.31
  028c9 f2 0f 58 44 06 
        e8               addsd -24(%rsi,%rax), %xmm0            #../src/feautrier.cpp:350.31
  028cf f2 0f 11 44 06 
        e8               movsd %xmm0, -24(%rsi,%rax)            #../src/feautrier.cpp:350.5
  028d5 48 83 c6 f0      addq $-16, %rsi                        #../src/feautrier.cpp:348.3
  028d9 4d 3b d1         cmpq %r9, %r10                         #../src/feautrier.cpp:348.3
  028dc 72 c0            jb ..B8.99 # Prob 63%                  #../src/feautrier.cpp:348.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r14 r15 xmm0 xmm2
..B8.100:                       # Preds ..B8.99
                                # Execution count [9.79e-01]
  028de 4a 8d 0c 55 01 
        00 00 00         lea 1(,%r10,2), %rcx                   #../src/feautrier.cpp:350.5
                                # LOE rdx rcx rbx rbp r8 r14 r15 xmm2
..B8.101:                       # Preds ..B8.100 ..B8.97
                                # Execution count [1.00e+00]
  028e6 48 ff c9         decq %rcx                              #../src/feautrier.cpp:350.5
  028e9 48 3b ca         cmpq %rdx, %rcx                        #../src/feautrier.cpp:348.3
  028ec 73 25            jae ..B8.103 # Prob 2%                 #../src/feautrier.cpp:348.3
                                # LOE rcx rbx rbp r8 r14 r15 xmm2
..B8.102:                       # Preds ..B8.101
                                # Execution count [9.79e-01]
  028ee 48 2b d9         subq %rcx, %rbx                        #../src/feautrier.cpp:350.5
  028f1 0f 28 c2         movaps %xmm2, %xmm0                    #../src/feautrier.cpp:350.31
  028f4 f2 41 0f 10 4c 
        d8 f8            movsd -8(%r8,%rbx,8), %xmm1            #../src/feautrier.cpp:350.19
  028fb f2 0f 58 44 dd 
        f0               addsd -16(%rbp,%rbx,8), %xmm0          #../src/feautrier.cpp:350.31
  02901 f2 0f 5e c8      divsd %xmm0, %xmm1                     #../src/feautrier.cpp:350.31
  02905 f2 41 0f 58 4c 
        d8 f0            addsd -16(%r8,%rbx,8), %xmm1           #../src/feautrier.cpp:350.31
  0290c f2 41 0f 11 4c 
        d8 f0            movsd %xmm1, -16(%r8,%rbx,8)           #../src/feautrier.cpp:350.5
                                # LOE rbp r8 r14 r15 xmm2
..B8.103:                       # Preds ..B8.101 ..B8.96 ..B8.102
                                # Execution count [1.00e+00]
  02913 f2 41 0f 10 40 
        08               movsd 8(%r8), %xmm0                    #../src/feautrier.cpp:353.17
  02919 f2 0f 58 54 24 
        40               addsd 64(%rsp), %xmm2                  #../src/feautrier.cpp:353.27[spill]
  0291f f2 0f 5e c2      divsd %xmm2, %xmm0                     #../src/feautrier.cpp:353.27
  02923 f2 41 0f 58 00   addsd (%r8), %xmm0                     #../src/feautrier.cpp:353.27
  02928 f2 41 0f 11 00   movsd %xmm0, (%r8)                     #../src/feautrier.cpp:353.3
  0292d 4d 85 ff         testq %r15, %r15                       #../src/feautrier.cpp:358.13
  02930 74 08            je ..B8.105 # Prob 32%                 #../src/feautrier.cpp:358.13
                                # LOE rbp r14 r15
..B8.104:                       # Preds ..B8.103
                                # Execution count [6.74e-01]
  02932 4c 89 ff         movq %r15, %rdi                        #../src/feautrier.cpp:358.3
  02935 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:358.3
                                # LOE rbp r14
..B8.105:                       # Preds ..B8.104 ..B8.103
                                # Execution count [1.00e+00]
  0293a 4d 85 f6         testq %r14, %r14                       #../src/feautrier.cpp:359.13
  0293d 74 08            je ..B8.107 # Prob 32%                 #../src/feautrier.cpp:359.13
                                # LOE rbp r14
..B8.106:                       # Preds ..B8.105
                                # Execution count [6.74e-01]
  0293f 4c 89 f7         movq %r14, %rdi                        #../src/feautrier.cpp:359.3
  02942 e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:359.3
                                # LOE rbp
..B8.107:                       # Preds ..B8.106 ..B8.105
                                # Execution count [6.74e-01]
  02947 48 89 ef         movq %rbp, %rdi                        #../src/feautrier.cpp:360.3
  0294a e8 fc ff ff ff  #       operator delete[](void *)
        call      _ZdaPv                                        #../src/feautrier.cpp:360.3
                                # LOE
..B8.108:                       # Preds ..B8.107
                                # Execution count [1.00e+00]
  0294f 33 c0            xorl %eax, %eax                        #../src/feautrier.cpp:363.10
  02951 48 83 c4 68      addq $104, %rsp                        #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 56
	.cfi_restore 6
  02955 5d               popq %rbp                              #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 48
	.cfi_restore 3
  02956 5b               popq %rbx                              #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 40
	.cfi_restore 15
  02957 41 5f            popq %r15                              #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 32
	.cfi_restore 14
  02959 41 5e            popq %r14                              #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 24
	.cfi_restore 13
  0295b 41 5d            popq %r13                              #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 16
	.cfi_restore 12
  0295d 41 5c            popq %r12                              #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 8
  0295f c3               ret                                    #../src/feautrier.cpp:363.10
	.cfi_def_cfa_offset 160
	.cfi_offset 3, -48
	.cfi_offset 6, -56
	.cfi_offset 12, -16
	.cfi_offset 13, -24
	.cfi_offset 14, -32
	.cfi_offset 15, -40
                                # LOE
..B8.109:                       # Preds ..B8.46
                                # Execution count [6.72e-02]: Infreq
  02960 48 8d 04 dd 00 
        00 00 00         lea (,%rbx,8), %rax                    #../src/feautrier.cpp:300.5
  02968 49 8d 0c df      lea (%r15,%rbx,8), %rcx                #../src/feautrier.cpp:300.5
  0296c 0f 84 e3 fc ff 
        ff               je ..B8.78 # Prob 50%                  #../src/feautrier.cpp:298.13
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.110:                       # Preds ..B8.109
                                # Execution count [3.36e-02]: Infreq
  02972 48 83 7c 24 18 
        00               cmpq $0, 24(%rsp)                      #../src/feautrier.cpp:310.30[spill]
  02978 0f 89 27 fd ff 
        ff               jns ..B8.80 # Prob 50%                 #../src/feautrier.cpp:310.30
  0297e e9 0e fe ff ff   jmp ..B8.91 # Prob 100%                #../src/feautrier.cpp:310.30
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.113:                       # Preds ..B8.88 ..B8.87
                                # Execution count [1.11e+00]: Infreq
  02983 4d 89 e9         movq %r13, %r9                         #../src/feautrier.cpp:315.3
  02986 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:315.3
  02989 49 c1 e9 3f      shrq $63, %r9                          #../src/feautrier.cpp:315.3
  0298d ba 01 00 00 00   movl $1, %edx                          #../src/feautrier.cpp:315.3
  02992 4d 03 cd         addq %r13, %r9                         #../src/feautrier.cpp:226.5
  02995 45 33 db         xorl %r11d, %r11d                      #../src/feautrier.cpp:317.5
  02998 49 d1 f9         sarq $1, %r9                           #../src/feautrier.cpp:226.5
  0299b 4d 85 c9         testq %r9, %r9                         #../src/feautrier.cpp:315.3
  0299e 76 2f            jbe ..B8.117 # Prob 10%                #../src/feautrier.cpp:315.3
                                # LOE rax rdx rcx rbx rbp r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B8.114:                       # Preds ..B8.113
                                # Execution count [1.00e+00]: Infreq
  029a0 48 8b 74 24 20   movq 32(%rsp), %rsi                    #../src/feautrier.cpp:317.5[spill]
  029a5 4b 8d 14 e0      lea (%r8,%r12,8), %rdx                 #../src/feautrier.cpp:317.5
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B8.115:                       # Preds ..B8.115 ..B8.114
                                # Execution count [2.78e+00]: Infreq
  029a9 49 8b 3c 33      movq (%r11,%rsi), %rdi                 #../src/feautrier.cpp:317.17
  029ad 49 ff c2         incq %r10                              #../src/feautrier.cpp:315.3
  029b0 49 89 3c 13      movq %rdi, (%r11,%rdx)                 #../src/feautrier.cpp:317.5
  029b4 49 8b 7c 33 08   movq 8(%r11,%rsi), %rdi                #../src/feautrier.cpp:317.17
  029b9 49 89 7c 13 08   movq %rdi, 8(%r11,%rdx)                #../src/feautrier.cpp:317.5
  029be 49 83 c3 10      addq $16, %r11                         #../src/feautrier.cpp:315.3
  029c2 4d 3b d1         cmpq %r9, %r10                         #../src/feautrier.cpp:315.3
  029c5 72 e2            jb ..B8.115 # Prob 64%                 #../src/feautrier.cpp:315.3
                                # LOE rax rdx rcx rbx rbp rsi r8 r9 r10 r11 r12 r13 r14 r15 xmm2
..B8.116:                       # Preds ..B8.115
                                # Execution count [1.00e+00]: Infreq
  029c7 4a 8d 14 55 01 
        00 00 00         lea 1(,%r10,2), %rdx                   #../src/feautrier.cpp:317.5
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.117:                       # Preds ..B8.116 ..B8.113
                                # Execution count [1.11e+00]: Infreq
  029cf 48 ff ca         decq %rdx                              #../src/feautrier.cpp:317.5
  029d2 49 3b d5         cmpq %r13, %rdx                        #../src/feautrier.cpp:315.3
  029d5 0f 83 b6 fd ff 
        ff               jae ..B8.91 # Prob 10%                 #../src/feautrier.cpp:315.3
                                # LOE rax rdx rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.118:                       # Preds ..B8.117
                                # Execution count [1.00e+00]: Infreq
  029db 48 8b 74 24 20   movq 32(%rsp), %rsi                    #../src/feautrier.cpp:317.17[spill]
  029e0 4f 8d 14 e0      lea (%r8,%r12,8), %r10                 #../src/feautrier.cpp:317.5
  029e4 f2 0f 10 59 f8   movsd -8(%rcx), %xmm3                  #../src/feautrier.cpp:342.28
  029e9 4c 8b 0c d6      movq (%rsi,%rdx,8), %r9                #../src/feautrier.cpp:317.17
  029ed 4d 89 0c d2      movq %r9, (%r10,%rdx,8)                #../src/feautrier.cpp:317.5
  029f1 e9 a0 fd ff ff   jmp ..B8.92 # Prob 100%                #../src/feautrier.cpp:317.5
                                # LOE rax rbx rbp r8 r12 r13 r14 r15 xmm2 xmm3
..B8.121:                       # Preds ..B8.49 ..B8.51 ..B8.53
                                # Execution count [0.00e+00]: Infreq
  029f6 45 33 d2         xorl %r10d, %r10d                      #../src/feautrier.cpp:274.5
  029f9 49 83 f9 01      cmpq $1, %r9                           #../src/feautrier.cpp:274.5
  029fd 0f 82 bd fb ff 
        ff               jb ..B8.74 # Prob 50%                  #../src/feautrier.cpp:274.5
                                # LOE rbx rbp r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.122:                       # Preds ..B8.66 ..B8.121
                                # Execution count [0.00e+00]: Infreq
  02a03 33 f6            xorl %esi, %esi                        #../src/feautrier.cpp:274.5
  02a05 e9 62 fb ff ff   jmp ..B8.70 # Prob 100%                #../src/feautrier.cpp:274.5
                                # LOE rbx rbp rsi r8 r9 r10 r12 r13 r14 r15 xmm1 xmm2
..B8.123:                       # Preds ..B8.47
                                # Execution count [6.05e-02]: Infreq
  02a0a 48 8d 04 dd 00 
        00 00 00         lea (,%rbx,8), %rax                    #../src/feautrier.cpp:300.5
  02a12 49 8d 0c df      lea (%r15,%rbx,8), %rcx                #../src/feautrier.cpp:300.5
  02a16 0f 84 39 fc ff 
        ff               je ..B8.78 # Prob 50%                  #../src/feautrier.cpp:298.13
  02a1c e9 84 fc ff ff   jmp ..B8.80 # Prob 100%                #../src/feautrier.cpp:298.13
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.124:                       # Preds ..B8.21 ..B8.25 ..B8.23
                                # Execution count [0.00e+00]: Infreq
  02a21 33 f6            xorl %esi, %esi                        #../src/feautrier.cpp:264.5
  02a23 48 83 f9 01      cmpq $1, %rcx                          #../src/feautrier.cpp:264.5
  02a27 0f 82 4a f8 ff 
        ff               jb ..B8.46 # Prob 50%                  #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp rsi r8 r12 r13 r14 r15 xmm1 xmm2
..B8.125:                       # Preds ..B8.38 ..B8.124
                                # Execution count [0.00e+00]: Infreq
  02a2d 33 ff            xorl %edi, %edi                        #../src/feautrier.cpp:264.5
  02a2f e9 e6 f7 ff ff   jmp ..B8.42 # Prob 100%                #../src/feautrier.cpp:264.5
                                # LOE rdx rcx rbx rbp rsi rdi r8 r12 r13 r14 r15 xmm1 xmm2
..B8.126:                       # Preds ..B8.17
                                # Execution count [0.00e+00]: Infreq
  02a34 f2 0f 10 15 fc 
        ff ff ff         movsd .L_2il0floatpacket.62(%rip), %xmm2 #../src/feautrier.cpp:283.19
  02a3c 4d 85 ed         testq %r13, %r13                       #../src/feautrier.cpp:272.12
  02a3f 7e 17            jle ..B8.128 # Prob 16%                #../src/feautrier.cpp:272.12
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.127:                       # Preds ..B8.126
                                # Execution count [0.00e+00]: Infreq
  02a41 f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:276.19
  02a49 49 8d 44 24 ff   lea -1(%r12), %rax                     #../src/feautrier.cpp:310.15
  02a4e 48 89 44 24 18   movq %rax, 24(%rsp)                    #../src/feautrier.cpp:310.15[spill]
  02a53 e9 32 f8 ff ff   jmp ..B8.48 # Prob 100%                #../src/feautrier.cpp:310.15
                                # LOE rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.128:                       # Preds ..B8.126
                                # Execution count [3.23e-02]: Infreq
  02a58 48 8d 04 dd 00 
        00 00 00         lea (,%rbx,8), %rax                    #../src/feautrier.cpp:300.5
  02a60 49 8d 0c df      lea (%r15,%rbx,8), %rcx                #../src/feautrier.cpp:300.5
  02a64 4d 85 e4         testq %r12, %r12                       #../src/feautrier.cpp:288.14
  02a67 75 16            jne ..B8.130 # Prob 50%                #../src/feautrier.cpp:288.14
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
..B8.129:                       # Preds ..B8.128
                                # Execution count [1.62e-02]: Infreq
  02a69 f2 0f 10 0d fc 
        ff ff ff         movsd .L_2il0floatpacket.60(%rip), %xmm1 #../src/feautrier.cpp:291.12
  02a71 48 c7 44 24 18 
        ff ff ff ff      movq $-1, 24(%rsp)                     #../src/feautrier.cpp:310.15[spill]
  02a7a e9 90 fb ff ff   jmp ..B8.75 # Prob 100%                #../src/feautrier.cpp:310.15
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm1 xmm2
..B8.130:                       # Preds ..B8.128
                                # Execution count [1.62e-02]: Infreq
  02a7f 49 8d 54 24 ff   lea -1(%r12), %rdx                     #../src/feautrier.cpp:310.15
  02a84 48 89 54 24 18   movq %rdx, 24(%rsp)                    #../src/feautrier.cpp:310.15[spill]
  02a89 e9 ba fb ff ff   jmp ..B8.76 # Prob 100%                #../src/feautrier.cpp:310.15
  02a8e 66 90            .align    16,0x90
                                # LOE rax rcx rbx rbp r8 r12 r13 r14 r15 xmm2
	.cfi_endproc
# mark_end;
	.type	_Z9feautrierlPdS_lS_S_S_,@function
	.size	_Z9feautrierlPdS_lS_S_S_,.-_Z9feautrierlPdS_lS_S_S_
	.data
# -- End  _Z9feautrierlPdS_lS_S_S_
	.text
# -- Begin  __sti__$E
	.text
# mark_begin;
       .align    16,0x90
# --- __sti__$E()
__sti__$E:
..B9.1:                         # Preds ..B9.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value___sti__$E.292:
..L293:
                                                        #
  02a90 56               pushq %rsi                             #
	.cfi_def_cfa_offset 16
  02a91 bf 00 00 00 00   movl $_ZN17_INTERNALd5350bc0St8__ioinitE, %edi #/usr/include/c++/5/iostream:74.25
..___tag_value___sti__$E.295:
  02a96 e8 fc ff ff ff  #       std::ios_base::Init::Init(std::ios_base::Init *)
        call      _ZNSt8ios_base4InitC1Ev                       #/usr/include/c++/5/iostream:74.25
..___tag_value___sti__$E.296:
                                # LOE rbx rbp r12 r13 r14 r15
..B9.2:                         # Preds ..B9.1
                                # Execution count [1.00e+00]
  02a9b bf 00 00 00 00   movl $_ZNSt8ios_base4InitD1Ev, %edi    #/usr/include/c++/5/iostream:74.25
  02aa0 be 00 00 00 00   movl $_ZN17_INTERNALd5350bc0St8__ioinitE, %esi #/usr/include/c++/5/iostream:74.25
  02aa5 ba 00 00 00 00   movl $__dso_handle, %edx               #/usr/include/c++/5/iostream:74.25
  02aaa 48 83 c4 08      addq $8, %rsp                          #/usr/include/c++/5/iostream:74.25
	.cfi_def_cfa_offset 8
  02aae e9 fc ff ff ff  #       __cxa_atexit()
        jmp       __cxa_atexit                                  #/usr/include/c++/5/iostream:74.25
  02ab3 0f 1f 44 00 00 
        0f 1f 84 00 00 
        00 00 00         .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	__sti__$E,@function
	.size	__sti__$E,.-__sti__$E
	.data
# -- End  __sti__$E
	.bss
	.align 1
_ZN17_INTERNALd5350bc0St8__ioinitE:
	.type	_ZN17_INTERNALd5350bc0St8__ioinitE,@object
	.size	_ZN17_INTERNALd5350bc0St8__ioinitE,1
	.space 1	# pad
	.section .rodata, "a"
	.align 32
__$U7:
	.long	1701275973
	.long	1295661678
	.long	1631744097
	.long	1144808819
	.long	1986622053
	.long	539780197
	.long	976895536
	.long	1114661197
	.long	677737313
	.long	1701275973
	.long	1765423726
	.long	1919251566
	.long	980181358
	.long	1852793658
	.long	1769236836
	.long	1818324591
	.long	2019900476
	.long	1936028272
	.long	1852795251
	.long	1159736382
	.long	1852139369
	.long	1852389946
	.long	1852990836
	.long	976907361
	.long	1767993972
	.long	1144812404
	.long	1986622053
	.long	977167461
	.long	1633899322
	.long	544366956
	.long	1663052842
	.long	1953721967
	.long	1734952224
	.long	976907877
	.long	1702129257
	.long	1818324594
	.long	1920219706
	.long	1937008993
	.long	1919239228
	.long	1684371049
	.long	1396324926
	.long	1634492771
	.long	1042948210
	.long	2037660218
	.long	539780464
	.long	1735290732
	.long	1869357100
	.long	539584366
	.long	1953068891
	.long	1698963560
	.long	1702259058
	.long	540876900
	.long	1701275973
	.long	1379547758
	.long	1161586277
	.long	1852139369
	.long	1632451130
	.long	2020176500
	.long	1970234428
	.long	744844386
	.long	741420320
	.long	741420320
	.long	539766816
	.long	539767085
	.long	742273325
	.long	539766816
	.long	1701275973
	.long	1329216110
	.long	1919251573
	.long	1769108563
	.long	758932836
	.long	1564360241
	.byte	0
	.type	__$U7,@object
	.size	__$U7,289
	.space 31, 0x00 	# pad
	.align 32
__$U9:
	.long	1701275973
	.long	1396324974
	.long	1684632180
	.long	1331641445
	.long	1919251573
	.long	1769108563
	.long	1950442852
	.long	1886220099
	.long	1415933033
	.long	744844649
	.long	1850302240
	.long	1400005998
	.long	1684632180
	.long	1131692389
	.long	1768975727
	.long	1767138668
	.long	977167725
	.long	1920226106
	.long	677733481
	.long	1735290732
	.long	1869357100
	.long	539584366
	.long	1953068891
	.long	1852383336
	.long	1331634292
	.long	1919251573
	.long	1769108563
	.long	1950442852
	.long	1886220099
	.long	1415933033
	.long	543518057
	.long	825040957
	.long	1852383276
	.long	1230970996
	.long	1919250030
	.long	1769108563
	.long	1950442852
	.long	1886220099
	.long	1415933033
	.long	543518057
	.long	1563435069
	.byte	0
	.type	__$U9,@object
	.size	__$U9,165
	.space 27, 0x00 	# pad
	.align 32
__$U5:
	.long	1701275973
	.long	1765423726
	.long	1919251566
	.long	980181358
	.long	1634890810
	.long	1014199401
	.long	1769104708
	.long	1046766966
	.long	1666398778
	.long	1918987361
	.long	1766139424
	.long	980313447
	.long	1852130362
	.long	1866687859
	.long	1936090725
	.long	1702060354
	.long	1919239228
	.long	1684371049
	.long	1043406892
	.long	1886337594
	.long	1952543333
	.long	690516591
	.long	1852795944
	.long	1814047847
	.long	694644335
	.long	1769429792
	.long	1142974580
	.long	1986622053
	.long	1025533029
	.long	1734952224
	.long	976907877
	.long	1013343570
	.long	1701275973
	.long	1295661678
	.long	1769108577
	.long	1868840056
	.long	1701601909
	.long	825040940
	.long	825040940
	.long	741351468
	.long	741420320
	.long	1043410208
	.long	741351468
	.long	1734952224
	.long	976907877
	.long	1702131023
	.long	1920226162
	.long	1013277801
	.long	1044263213
	.word	93
	.type	__$U5,@object
	.size	__$U5,198
	.space 10, 0x00 	# pad
	.align 16
.L_2il0floatpacket.61:
	.long	0x00000000,0x40000000,0x00000000,0x40000000
	.type	.L_2il0floatpacket.61,@object
	.size	.L_2il0floatpacket.61,16
	.align 8
.L_2il0floatpacket.60:
	.long	0x00000000,0x40000000
	.type	.L_2il0floatpacket.60,@object
	.size	.L_2il0floatpacket.60,8
	.align 8
.L_2il0floatpacket.62:
	.long	0x00000000,0x3ff00000
	.type	.L_2il0floatpacket.62,@object
	.size	.L_2il0floatpacket.62,8
	.align 1
__$U2:
	.long	1684631414
	.long	1766140448
	.long	980313447
	.long	1953392954
	.long	1634628197
	.long	1631205996
	.long	1852270956
	.long	1834968165
	.long	1869376609
	.long	1853171811
	.long	1852270963
	.long	1814062181
	.long	694644335
	.byte	0
	.type	__$U2,@object
	.size	__$U2,53
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
.L_2__STRING.17:
	.long	1835104357
	.long	1600482416
	.long	1635017060
	.long	1634035247
	.long	1769108597
	.long	774992485
	.long	7633012
	.type	.L_2__STRING.17,@object
	.size	.L_2__STRING.17,28
	.align 4
.L_2__STRING.0:
	.long	1701669236
	.long	540876832
	.byte	0
	.type	.L_2__STRING.0,@object
	.size	.L_2__STRING.0,9
	.space 3, 0x00 	# pad
	.align 4
.L_2__STRING.4:
	.long	1701736041
	.long	1920226162
	.long	1046832233
	.long	639643709
	.long	1970216998
	.long	1400006004
	.long	1684632180
	.long	809320037
	.byte	0
	.type	.L_2__STRING.4,@object
	.size	.L_2__STRING.4,33
	.space 3, 0x00 	# pad
	.align 4
.L_2__STRING.6:
	.long	544698226
	.long	807419198
	.long	539371040
	.long	544698226
	.long	1869750332
	.long	690516855
	.long	539371040
	.long	543977315
	.long	807419198
	.long	539371040
	.long	543977315
	.long	1868767292
	.long	690516844
	.byte	0
	.type	.L_2__STRING.6,@object
	.size	.L_2__STRING.6,53
	.section .rodata.str1.32, "aMS",@progbits,1
	.align 32
.L_2__STRING.8:
	.long	1952539688
	.long	1920225377
	.long	540884256
	.long	2082482480
	.long	539500668
	.long	1937207154
	.long	540884512
	.long	640032816
	.long	1867655200
	.long	1950446455
	.long	1886220099
	.long	1415933033
	.long	543518057
	.long	1142963517
	.long	1835101817
	.long	2082497385
	.long	1867653244
	.long	1950446455
	.long	1886220099
	.long	1415933033
	.long	543518057
	.long	1914715453
	.long	695433071
	.long	539371040
	.long	1936486243
	.long	540884512
	.long	640032816
	.long	1866672160
	.long	1950446444
	.long	1886220099
	.long	1415933033
	.long	543518057
	.long	1142963517
	.long	1835101817
	.long	2082497385
	.long	1866670204
	.long	1950446444
	.long	1886220099
	.long	1415933033
	.long	543518057
	.long	1663057213
	.long	695430255
	.word	41
	.type	.L_2__STRING.8,@object
	.size	.L_2__STRING.8,170
	.space 22, 0x00 	# pad
	.align 32
.L_2__STRING.9:
	.long	1836017711
	.long	1919299429
	.long	1919247461
	.long	1143958377
	.long	1651535730
	.long	1093630063
	.long	1869771891
	.long	1734429999
	.long	1953786226
	.long	1869426533
	.long	1701606756
	.long	1632776051
	.long	1952541028
	.long	1415935593
	.long	1936613746
	.long	796026214
	.long	795046515
	.long	1701275973
	.long	1920151406
	.long	1866674019
	.long	1294951794
	.long	1631744097
	.long	1747871091
	.byte	0
	.type	.L_2__STRING.9,@object
	.size	.L_2__STRING.9,93
	.space 3, 0x00 	# pad
	.align 32
.L_2__STRING.5:
	.long	1836017711
	.long	1919299429
	.long	1919247461
	.long	1143958377
	.long	1651535730
	.long	1093630063
	.long	1869771891
	.long	1734429999
	.long	1953786226
	.long	1869426533
	.long	1701606756
	.long	1632776051
	.long	1952541028
	.long	1415935593
	.long	1936613746
	.long	796026214
	.long	795046515
	.long	1701275973
	.long	1920151406
	.long	1866674019
	.long	1395615090
	.long	1684632180
	.long	6827621
	.type	.L_2__STRING.5,@object
	.size	.L_2__STRING.5,92
	.space 4, 0x00 	# pad
	.align 32
.L_2__STRING.14:
	.long	2053731112
	.long	909196389
	.long	545029152
	.long	1685353256
	.long	1769159226
	.long	1952408954
	.long	1936028200
	.long	695495797
	.long	691417381
	.long	691027261
	.long	539371040
	.long	1937330978
	.long	661480820
	.long	1634541683
	.long	1668246636
	.long	1952805408
	.long	1701737077
	.long	1851859044
	.long	1634628896
	.long	1852270956
	.long	1881171045
	.long	1953393007
	.long	539914853
	.long	1886220099
	.long	543517801
	.long	1752459639
	.long	1195984160
	.long	1298091589
	.long	1330400321
	.long	1279352643
	.long	1145128274
	.long	1279352665
	.long	1162757961
	.long	540032324
	.long	1713401716
	.long	1651272801
	.long	543908705
	.long	1746956148
	.long	1835298401
	.long	543515745
	.long	1734962273
	.long	1830839406
	.long	1919905125
	.long	1818304633
	.long	1633906540
	.long	779251572
	.word	34
	.type	.L_2__STRING.14,@object
	.size	.L_2__STRING.14,186
	.space 6, 0x00 	# pad
	.align 32
.L_2__STRING.15:
	.long	1836017711
	.long	1919299429
	.long	1919247461
	.long	1143958377
	.long	1651535730
	.long	1093630063
	.long	1869771891
	.long	1734429999
	.long	1953786226
	.long	1869426533
	.long	1701606756
	.long	1632776051
	.long	1952541028
	.long	1415935593
	.long	1936613746
	.long	796026214
	.long	795046515
	.long	1701275973
	.long	1920151406
	.long	1866674019
	.long	1966040434
	.long	795634036
	.long	1869440333
	.long	1747876210
	.byte	0
	.type	.L_2__STRING.15,@object
	.size	.L_2__STRING.15,97
	.space 31, 0x00 	# pad
	.align 32
.L_2__STRING.7:
	.long	1836017711
	.long	1919299429
	.long	1919247461
	.long	1143958377
	.long	1651535730
	.long	1093630063
	.long	1869771891
	.long	1734429999
	.long	1953786226
	.long	1869426533
	.long	1701606756
	.long	1632776051
	.long	1952541028
	.long	1415935593
	.long	1936613746
	.long	796026214
	.long	795046515
	.long	1701275973
	.long	1920151406
	.long	1866674019
	.long	1143956850
	.long	1702063717
	.long	1717923651
	.long	1631744870
	.long	1747871091
	.byte	0
	.type	.L_2__STRING.7,@object
	.size	.L_2__STRING.7,101
	.section .bss._ZZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0, "waG",@nobits,_ZZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0,comdat
	.align 16
	.weak _ZZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0
_ZZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0:
	.type	_ZZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0,@object
	.size	_ZZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0,16
	.space 16	# pad
	.section .bss._ZGVZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0, "waG",@nobits,_ZGVZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0,comdat
	.align 8
	.weak _ZGVZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0
_ZGVZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0:
	.type	_ZGVZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0,@object
	.size	_ZGVZN5Eigen8internal4pexpI7__m128dEET_RKS3_E10p4i_1023_0,8
	.space 8	# pad
	.section .bss._ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes, "waG",@nobits,_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes,comdat
	.align 8
	.weak _ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes
_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes:
	.type	_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes,@object
	.size	_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes,24
	.space 24	# pad
	.section .bss._ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes, "waG",@nobits,_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes,comdat
	.align 8
	.weak _ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes
_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes:
	.type	_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes,@object
	.size	_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes,8
	.space 8	# pad
	.section .ctors, "wa"
	.align 8
__init_0:
	.type	__init_0,@object
	.size	__init_0,8
	.quad	__sti__$E
	.data
	.hidden __dso_handle
	.set _ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsrNS9_6traitsIT_EE9AlignmentLi0EEPvE4typeE,_ZNK5Eigen7MapBaseINS_3RefINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0ENS_11OuterStrideILin1EEEEELi0EE11checkSanityIS6_EEvNS_8internal9enable_ifIXeqsr5Eigen8internal6traitsIT_EE9AlignmentLi0EEPvE4typeE
# mark_proc_addr_taken __sti__$E;
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
