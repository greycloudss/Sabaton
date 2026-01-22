#!/usr/bin/env bash
set -euo pipefail

AL_LT="AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ"
AL_EN="ABCDEFGHIKLMNOPQRSTUVWXYZ"
AL_ENIGMA="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

OUT="all_deciphers.txt"
: > "$OUT"

add() {
    echo "$1" >> "$OUT"
}

###############################################################################
# 67 VARIANTAS
###############################################################################
add "##### SCYTALE (67.1) #####"
add "# brute rows 1..29"
./sabaton.exe -decypher -scytale -alph "$AL_LT" "$(tr -d '[:space:]' <<'TXT'
PGUPIŠSAYBISPAMBĖPŲLVAIGUPIOTNOLUABĖIKSSUOSAAIIKTAIDŠŲŠARSAVATG
TXT
)" >> "$OUT"
add ""

add "##### TRANSPOSITION (67.2 key=DUKRA) #####"
./sabaton.exe -decypher -transposition -alph "$AL_LT" -frag "DUKRA" "$(tr -d '[:space:]' <<'TXT'
IGEOVIALNUIŠEVYUIUAAPSRUJKUIĮĄAINTODSIONODGSVATAIMAAAOEKMSVSKDRSĖIASNRIITOTVLKRIITEIRAŽJSAMSEAŠMEŽIDGKGRIIJAIPIYKŽDIOPUD
TXT
)" >> "$OUT"
add ""

add "##### TRANSPOSITION (67.3 len=6 brute) #####"
./sabaton.exe -decypher -transposition -alph "$AL_LT" -frag 6 "$(tr -d '[:space:]' <<'TXT'
UOEAKIKKRĮŪIBŠOSĖDITBPNBAOŪŲARJOAODTNTŠVOETTMŽEKĘADMULARTLTŠOEKOPKAINLTIAEĮIILSMETITJAAIREĄENKŽVSIUIDS
TXT
)" >> "$OUT"
add ""

add "##### FLEISSNER (67.4 mask [0,3,4,5,8]) #####"
./sabaton.exe -decypher -fleissner -frag "4;0001110010000000" "ŠMISUOŠAŠLSSUAMKIDAĖSSSUOUŽALOSSOAIMARPSGŠIPUPIAČLAEVOUASIISUULKŠUGBĖSRŽSIĘIŠLTĮAŲKESNRŠIUTŲPKŠAŲAITELŠAILUASŲAU" >> "$OUT"
add ""

add "##### BIFID (67.5 key KAIMAS period 5, EN) #####"
./sabaton.exe -decypher -bifid -alph "$AL_EN" -frag "KAIMAS;5" "$(tr -d '[:space:]' <<'TXT'
AKIRAMTFUNICDXRMVLGNEKMQLKMLOIKBXXBNEZMGKKLWOLKMZLPMRVOGTSYZRKYVVBENLOBIIOHBAVDOMGLVNOXSYR
TXT
)" >> "$OUT"
add ""

###############################################################################
# 56 VARIANTAS
###############################################################################
add "##### AFFINE CAESAR (56.1 prefix ŽOD) #####"
./sabaton.exe -decypher -affineCaesar -alph "$AL_LT" -frag "ŽOD" "$(tr -d '[:space:]' <<'TXT'
EĄAFYĘŽPYIFKŽJYMFYŲEDPŽAČRJFRFYKPYPDĘIPYIFMDPCŲCJIŲ
TXT
)" >> "$OUT"
add ""

add "##### AFFINE CAESAR (56.1 brute all keys) #####"
./sabaton.exe -decypher -affineCaesar -enhancedBrute -alph "$AL_LT" "$(tr -d '[:space:]' <<'TXT'
EĄAFYĘŽPYIFKŽJYMFYŲEDPŽAČRJFRFYKPYPDĘIPYIFMDPCŲCJIŲ
TXT
)" >> "$OUT"
add ""

add "##### AFFINE CAESAR (56.2 first letter T) #####"
./sabaton.exe -decypher -affineCaesar -alph "$AL_LT" -frag "T" "$(tr -d '[:space:]' <<'TXT'
ĮKOUČHKHKŪŪAOKHMUČĘGKŪŪKVUČLKRČHŠHZFUČĮKRČHOPĮČKUGKŪOKČ
TXT
)" >> "$OUT"
add ""

add "##### AFFINE CAESAR (56.2 brute) #####"
./sabaton.exe -decypher -affineCaesar -enhancedBrute -alph "$AL_LT" "$(tr -d '[:space:]' <<'TXT'
ĮKOUČHKHKŪŪAOKHMUČĘGKŪŪKVUČLKRČHŠHZFUČĮKRČHOPĮČKUGKŪOKČ
TXT
)" >> "$OUT"
add ""

add "##### AFFINE CAESAR (56.3 brute) #####"
./sabaton.exe -decypher -affineCaesar -enhancedBrute -alph "$AL_LT" "$(tr -d '[:space:]' <<'TXT'
NEČĘIRČLĖŲČEŠDČTČRAOĘHDGRKŲČLRĖĄKDČCČĘUŲĘFĘGIĖLĖŪČIŲ
TXT
)" >> "$OUT"
add ""

add "##### HILL (56.4 key [9,24,20,25]) #####"
./sabaton.exe -decypher -hill -alph "$AL_LT" -frag "9,24,20,25" "$(tr -d '[:space:]' <<'TXT'
VŠCLČĮLČĘEEKUEHĖŽŪSŪJĖTZČVOBIŽĄŠPIAŪZYBČNDOFŠŽ
TXT
)" >> "$OUT"
add ""

###############################################################################
# 15 VARIANTAS
###############################################################################
add "##### VIGENERE (15.1 key DUKRA) #####"
./sabaton.exe -decypher -vigenere -alph "$AL_LT" -frag "DUKRA" "$(tr -d '[:space:]' <<'TXT'
ZUOBAŽFKKPTGRĘITHŪEOŪIRJŲUUKBŠPBŽGKDŽDĘAŠAFGSTKMBTDĮRIOGYŽGKLPĄRUPEKĖŲUAEĖALĮHJAŲĖFKMOEMMTAJKĮTTSŪJOPEKHAZJKĮTTSŪEALJBMSALKKLĮLĘBSZABGSUHKFEZPČIBLNČĮBŽSČHAPUEJOZEPRBDKŽRUŪOPBDČEKKSLOFBKŪHGĖUŪNRŠĖOEBRILKKHSPUŪUIDSŪĖALGGIIHJKJVLKFBNTNŪETDEARSĘOĮGŽLĮČĖAPUPĮOREKIITYEŠIZUCŪPDLFGVL
TXT
)" >> "$OUT"
add ""

add "##### VIGENERE (15.2 prefix MEN 3-12) #####"
./sabaton.exe -decypher -vigenere -alph "$AL_LT" -frag "prefix:MEN|3-12" "$(tr -d '[:space:]' <<'TXT'
FAĖKSCHIEAIGZŠĮDAFMGŽKUĘMHČKIĘEŽĮEILSEĮUISITTŪPLFCNĄAGSNCLGCHUCŪAEMADKNHĘŪŽUIFHGĘKĮIŲĮĄIGŪEŽĮYĘSOIŪVCFAGFIVBVKKRTZAGFIVBŠTCKLFNGŪPSĖIŽUOEZŠĮĘDOZHTĖIFHRTFUĮJEČKNŪZTŽČLSZRFČDRZURŪFVĖĖČKTHĖUŠZKĖUVFČDŽFIĖFIGHĘKKLSDHOĘALCYČRIŪEVŽĘEIHEMEFAOMSNCDSJMTFOYCEŽAUOMHŽCAYZHKŪDVGMDFAIĖUDRTŽĄUČKIĘI
TXT
)" >> "$OUT"
add ""

add "##### VIGENERE (15.3 crack 3-12) #####"
./sabaton.exe -decypher -vigenere -alph "$AL_LT" -frag "crack:3-12" "$(tr -d '[:space:]' <<'TXT'
IBEIHOZBTHOCRACPZBNSKMBOYOBĘISNUGPHŽGŲĮINUŪŽSOĘBEYKOŠUOLČRSKVKŪTHMŽRNVRBŽIPFZRIIVDJETDVGRDPŲJAĮMĘRMKFZIŲINBTEYŠHČKYRYFČCŠCĘAPJUFISJOGKVVĄŪIĮYBEAĖFĄBŲĄNYNIKVLBNCLCGTVKLBAĖLDNYĮVHBMMĮĖIIVMČBKĖVĖĮĖGSBGKVVĄGMVBĮBAĄŠYHYĘVUJOYRYBGCEUOIĘVUGNVŽUGLHĘYBPSOUMLŽMYIAŪLZĮAĖFJŽOĮJĖĘUHRPHLSŲABNŽJYYIGVĘKRŽBYĖIĄVŽNOEMYŲAĮZAMS MOĖHRSRUĖUĄVŽBRTCŠČVDŽRĮTSĄURUĄCKRNKGĘBKYYUFGSFHŪIĘRUBŠĘVČJPĮFNKRĘFARSKNBFOFFĄBUFVDJUKVUŠUOLUJIGLČBGCKYĘUGLČĮNCCZRSGCĮBNHIBEACMĘSSNONIETSMB ACĘYĘIYRYŠIĖFIKOKFNRNĄVĖĮKNKBGRTFĘSJS SDŪNCVUŠUOLĮBNHJUARSUUJOĮFYBĮĮCĄAIGFYBSLĮĖIISKĘČSŪFČŠTCKUKSEDŠGSCNŠIAOFĘRCCIBĮLSSZKSTCĘFIVĮYĮNVRĄŪĮYFHRIPBYNOĘVUH AFLČŽIGDZGŠŽĘYĘIIVDBEĘRUIEPSŽJAKZZKRCOŠIAOFĘGRCSBĮEĄVĖFAFVDHEĮOĖŲĖKFĄAAYŽČRNŪSBĘIGCDIESĮLBJSOUIMCYĮBNCPZGJVROĖPVNYJŪĮLVŪČCVKKVHNOĖISJYĮIĖCERTVYVZDŽOĄBUĘVUHAKĮUITCOĘRMĘLĄĮEĮUYJYOFIĖUCNYŲOYFZBTMĖUHOKCIBŲIVHČZŪŪUKIOFOFACPŠŪRCVĖĮISFUYG OFŽŲEGRFUIOFĘRVCRBĄIILĘŪZŽĮĖIIYĮOĘBŽĮYŲOĘCYFAYHČRDCVŽŪKKFĄBOCPDBVDOĘČMHMYĮĖĘVHRMPBUFIYĮBIIŪLČBUYŽĖNOKSEUISOYYPSOĘGVŽIYKVSYYFDŽYPELSSDČDSJYĮIYMČBEJSĮŲAĮVDŲUĮFDĮIVKFĖAKHĘŠŪKVCĘOGŠŠBRŪDVGSCĮYBPĖVĖEEOFČEALIYNĖEVDRNKBĖIŲČĮŪHACĮYŠOELYJSSFĄBAĮVĄEAINUĘIIFĄJAYMŽŪIYRČGLHMYĮSLSĮIAJSCBEJRĖEURJBAUYŪUMRŽIYKĮKVĄŪACPZBACFEEEO CČOOKZĮGDČŽĖNALŽOFOĮGDARČŪĘBPSODFALRJŠEKOĖHRSRYKKSBĘRIGCŲĖAGLAR
TXT
)" >> "$OUT"
add ""

add "##### VIGENERE (15.4 autokey VYRAS) #####"
./sabaton.exe -decypher -vigenere -alph "$AL_LT" -frag "auto:VYRAS" "$(tr -d '[:space:]' <<'TXT'
CZĮPVĖGKĄIŪĄPĄBZĄICBGHOUYŽŽOMĘTNĘZŲGŽSMMŪĘGŠFĖBRYZZBZVMPGYJZĖOYŲSYRĖŠHHBYHHBPOŪHYĘAJPY ZĮICZGČCVĖKFV OĖDFNOĄĄŽNAUĄCŽAOĘNRPUIŽTĄFARHPTSĘSĖDGSSOŪČYCHIČUĖMTFSOĘJPVINĘDVCGATULULTLŲEČVIJĖNCŪĖBICLŪRŪFLJRNŽSĄBNMKĮTNDKZDLĮBĮLĖJŪGDVDĖŽDŪLŲŠDNDLBDELČYŽŲLFYPPEFHPUMŪČĄFŽEMZCŠESZĖŠŪGGYĖŪĄVOŽĖSHNN
TXT
)" >> "$OUT"
add ""

###############################################################################
# 21 VARIANTAS (ENIGMA)
###############################################################################
add "##### ENIGMA (21.1 no reflector key [12,20]) #####"
./sabaton.exe -decypher -enigma -alph "$AL_ENIGMA" -frag "R1:8,13,24,18,9,0,7,14,10,11,19,25,4,17,12,21,15,3,22,2,20,16,23,1,6,5|R2:10,2,21,18,23,6,16,14,8,11,1,25,15,20,0,24,17,19,22,5,4,3,9,12,13,7|KEY:12,20" "MSBHYONOKXMPRQXAEFKXQRAOKASDKMFXVMZEJSURSTBGPOWAIGT" >> "$OUT"
add ""

add "##### ENIGMA (21.2 key0=19, brute key1, plain starts J) #####"
./sabaton.exe -decypher -enigma -alph "$AL_ENIGMA" -frag "R1:10,2,11,18,8,20,19,25,23,1,15,9,14,6,24,0,17,7,22,21,4,12,5,3,16,13|R2:14,2,7,20,18,9,19,25,23,1,13,17,22,5,3,0,24,8,21,10,11,12,15,4,6,16|KEY:19|PLAIN:J" "GLPWURKKTRJXNYXDBGYYPZZFJEHFMRXVGKSSXLVRACOENNERWSLA" >> "$OUT"
add ""

add "##### ENIGMA (21.3 with reflector key [7,21]) #####"
./sabaton.exe -decypher -enigma -alph "$AL_ENIGMA" -frag "R1:20,3,24,18,8,5,15,4,7,11,0,13,9,22,12,23,10,1,19,21,17,16,2,25,6,14|R2:8,13,24,18,9,0,7,14,10,11,19,25,4,17,12,21,15,3,22,2,20,16,23,1,6,5|REF:2,4,0,6,1,11,3,8,7,13,16,5,15,9,18,12,10,19,14,17,25,22,21,24,23,20|KEY:7,21" "WYNESGNEWDKOFUBWKGPRYMGXLCVJVMLLGOXUOYQAVFGFOILVYJFMVOPQ" >> "$OUT"
add ""

###############################################################################
# 18 VARIANTAS (FEISTEL)
###############################################################################
add "##### FEISTEL (18.1 f=3 keys [248,82,45]) #####"
./sabaton.exe -decypher -feistel -frag "f=3;k=[248,82,45]" "[[11, 64], [3, 91], [13, 67], [15, 71], [14, 71], [24, 91], [2, 73], [16, 89], [27, 72], [2, 65], [28, 92], [17, 83], [23, 73], [0, 65], [13, 67], [8, 79], [5, 73], [0, 65], [8, 94], [17, 91], [20, 81], [19, 85], [8, 65], [18, 74], [4, 77], [26, 93], [17, 72], [23, 72], [25, 89], [23, 74], [17, 72], [11, 90], [9, 92], [18, 73], [29, 76], [25, 83], [20, 81], [31, 72], [15, 66], [17, 83], [3, 68], [12, 92], [29, 89], [1, 79], [10, 65], [18, 84], [24, 91], [31, 88], [19, 78], [29, 76], [5, 65], [9, 75], [25, 91], [2, 65], [14, 88], [0, 73], [11, 89], [4, 73], [29, 89], [28, 95], [19, 84], [13, 68], [25, 77], [12, 95], [0, 65], [25, 92], [23, 81], [29, 95], [4, 67], [27, 72], [5, 95], [28, 95], [8, 65], [15, 89], [2, 71], [0, 73], [1, 68], [26, 75], [12, 67], [17, 83], [5, 92], [1, 68], [15, 66], [2, 65], [17, 81], [23, 64], [0, 69]]" >> "$OUT"
add ""

add "##### FEISTEL (18.2 f=1 keys [?,30] brute) #####"
./sabaton.exe -decypher -feistel -frag "f=1;k=[?,30]" "[[92, 6], [91, 4], [74, 11], [78, 9], [80, 26], [68, 14], [81, 26], [79, 8], [77, 26], [77, 26], [76, 26], [74, 19], [87, 24], [93, 3], [74, 6], [74, 2], [72, 22], [76, 28], [77, 26], [86, 31], [95, 16], [80, 25], [86, 9], [91, 4], [70, 7], [93, 6], [74, 5], [92, 23], [79, 10], [94, 19], [93, 23], [76, 22], [77, 26], [78, 7], [79, 14], [76, 26], [90, 17], [78, 15], [93, 7], [77, 26], [78, 7], [80, 25], [78, 28], [77, 18], [75, 28], [79, 10], [92, 7], [74, 14], [93, 6], [78, 7], [80, 25], [76, 26], [74, 25], [87, 24], [94, 17], [71, 6], [75, 9], [72, 22], [78, 108]]" >> "$OUT"
add ""

add "##### FEISTEL (18.3 f=3 keys [?,?] prefix KA brute) #####"
./sabaton.exe -decypher -feistel -frag "f=3;k=[?,?]" "[[65, 75], [67, 79], [82, 91], [95, 82], [65, 65], [76, 78], [88, 67], [64, 82], [64, 90], [76, 64], [74, 76], [89, 71], [64, 80], [90, 64], [76, 76], [77, 85], [95, 66], [82, 83], [81, 93], [84, 91], [74, 79], [95, 71], [80, 89], [88, 70], [78, 76], [88, 71], [69, 65], [94, 87], [69, 71], [78, 72], [65, 83], [73, 83], [70, 87], [91, 83], [70, 77], [94, 87], [68, 78], [82, 95], [83, 91], [75, 71], [94, 87], [72, 79], [70, 67], [95, 71], [90, 91], [76, 82], [70, 72], [12, 101]]" >> "$OUT"
add ""

###############################################################################
# 33 VARIANTAS (BLOCK MODES)
###############################################################################
add "##### BLOCK (33.1 ECB/CBC/CFB/CRT keys [217,108,80] f=2) #####"
./sabaton.exe -decypher -block -frag "f=2;[217,108,80]" "[[107, 188], [100, 185], [104, 174], [107, 188], [100, 185], [118, 169], [122, 165], [117, 166], [115, 164], [108, 168], [104, 174], [114, 160], [105, 188], [105, 174], [112, 173], [114, 174], [96, 176], [116, 169], [114, 173], [103, 176], [111, 172], [114, 191], [116, 164], [101, 178], [97, 174], [100, 177], [100, 182], [106, 176], [97, 168], [115, 164], [110, 189], [106, 179]]" >> "$OUT"
add ""

add "##### BLOCK (33.2 CBC same keys f=2) #####"
./sabaton.exe -decypher -block -frag "f=2;[217,108,80]" "[[87, 92], [51, 240], [199, 83], [34, 121], [71, 218], [174, 111], [78, 94], [61, 249], [201, 87], [61, 117], [83, 203], [163, 100], [67, 87], [38, 238], [214, 71], [52, 102], [85, 194], [178, 107], [65, 95], [33, 247], [211, 83], [51, 115], [82, 207], [171, 126], [67, 77], [41, 245], [218, 73], [51, 118], [65, 199], [169, 111], [65, 89], [32, 225], [213, 73], [38, 127], [66, 214], [184, 115], [85, 93], [58, 255], [223, 66], [59, 102], [95, 217], [171, 109], [74, 69], [35, 252], [195, 87], [39, 125], [70, 211], [168, 98], [76, 86], [45, 232], [196, 95], [33, 119], [83, 211], [178, 31]]" >> "$OUT"
add ""

add "##### BLOCK (33.3 CFB same keys f=2) #####"
./sabaton.exe -decypher -block -frag "f=2;[217,108,80]" "[[68, 73], [45, 244], [223, 77], [41, 97], [71, 217], [179, 110], [87, 82], [55, 253], [214, 88], [33, 126], [74, 205], [190, 120], [84, 69], [53, 228], [196, 94], [44, 116], [68, 192], [172, 119], [73, 89], [32, 247], [193, 88], [52, 113], [70, 205], [175, 123], [72, 75], [41, 226], [219, 87], [58, 105], [83, 215], [180, 104], [84, 91], [61, 242], [201, 73], [32, 117], [83, 217], [178, 97], [86, 95], [35, 225], [198, 79], [40, 98], [70, 203], [167, 120], [71, 83], [37, 239], [209, 76], [57, 117], [75, 209], [187, 101], [73, 91], [37, 239], [196, 95], [52, 117], [91, 208], [187, 112], [79, 82], [47, 225], [206, 81], [62, 125], [87, 222], [166, 119], [85, 69], [32, 227], [211, 76], [39, 107], [70, 178]]" >> "$OUT"
add ""

add "##### BLOCK (33.4 CRT same keys f=2) #####"
./sabaton.exe -decypher -block -frag "f=2;[217,108,80]" "[[173, 230], [176, 230], [191, 255], [190, 249], [184, 231], [178, 240], [183, 243], [173, 242], [162, 232], [164, 235], [182, 237], [184, 237], [168, 247], [162, 233], [182, 238], [188, 242], [184, 248], [172, 251], [180, 253], [186, 251], [183, 245], [175, 252], [168, 247], [168, 231], [176, 233], [181, 239], [166, 232], [182, 236], [186, 253], [184, 240], [178, 237]]" >> "$OUT"
add ""

###############################################################################
# 29 VARIANTAS (AES-V)
###############################################################################
add "##### AES-V (29.1 decrypt) #####"
./sabaton.exe -decypher -aes -frag "p:317|a:13|b:15|T:1,11,31,4|K:246,200,275,167|R:3" "[[229, 125, 291, 259], [128, 177, 11, 275], [17, 78, 282, 48], [233, 152, 195, 3], [281, 205, 193, 145], [106, 196, 76, 265], [287, 310, 239, 46], [238, 168, 179, 50], [182, 81, 93, 68], [139, 229, 243, 69], [246, 89, 231, 310], [310, 285, 122, 136], [314, 53, 242, 47], [281, 158, 85, 219]]" >> "$OUT"
add ""

add "##### AES-V (29.2 MITM K1,K2) #####"
./sabaton.exe -decypher -aes -frag "p:317|a:13|b:15|T:1,11,31,4|K1:?,179,178,226|K2:272,?,233,246|M:18,93,198,198|R:3" "[151,10,160,239]" >> "$OUT"
add ""

###############################################################################
# 60 VARIANTAS (STREAM)
###############################################################################
add "##### STREAM LFSR (60.1 brute prefix BE) #####"
./sabaton.exe -decypher -stream -frag "8;BE" "[234, 144, 51, 95, 200, 13, 206, 110, 2, 34, 31, 11, 228, 127, 161, 129, 239, 120, 84, 84, 20, 111, 121, 24, 142, 235, 208, 171, 141, 89, 157, 132, 27, 229, 157, 115, 113, 255, 79, 20, 213, 128, 207, 193, 24, 56, 168, 246, 54, 55, 94, 127, 250, 27]" >> "$OUT"
add ""

add "##### STREAM A5 (60.2) #####"
add "# Fill taps/state from 60.1 result then run:"
add "# ./sabaton.exe -decypher -a5 -frag \"TAPS_FROM_60_1\" \"[220, 224, 209, 48, 97, 234, 244, 167, 208, 8, 203, 164, 151, 221, 212, 202, 136, 214, 8, 35, 50, 186, 9, 19, 229, 121, 176, 155, 249, 119, 67, 82, 8, 100, 116, 18, 147, 239, 219, 168, 141, 91, 155, 149, 30, 249, 128, 120, 100, 238, 69, 12, 223, 133, 195, 194, 5, 36, 185, 236, 42, 33, 72, 127, 245, 29, 57, 253, 202, 3, 110, 142, 213, 102, 252, 236, 232, 23, 201, 46, 32, 16, 93, 200, 67, 248, 70, 83, 255, 182, 152, 25, 185, 183, 88, 63, 41, 230, 145, 38]\""
add ""

###############################################################################
# Remaining modules from assignment
###############################################################################
add "##### MERKLE (CPU) #####"
./sabaton.exe -decypher -merkle -alph "ABCDEFGHIJKLMNOPQRSTUVWXYZ " -frag "key:39342119,111996362,9301087,85349912,114247265,79246980,68969224,40465975|p:114537401|w1:811451" "[206647361,326360316,424226893,314791694,241010404,9301087]" >> "$OUT"
add ""

add "##### GRAHAM (CPU) #####"
./sabaton.exe -decypher -graham -alph "ABCDEFGHIJKLMNOPQRSTUVWXYZ " -frag "key:158474690964,197857591142,123672933023,16130333379,142979253486,151965468545,30534200386,127450405592|p:211936606955|w1:41985769523" "[495043978037,617076731681,774459852174]" >> "$OUT"
add ""

add "##### ELGAMAL #####"
./sabaton.exe -decypher -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -elgamal -frag "1;2;3;4" "[1,3584217634602882250414174591984384801148279487761588373233371085330221,2,3572561055637859460099057608692160179234251257895318983525239092556191,1627211721333001232207181324,1380067339047144425076529243235427669785561426931311250902916681307956,679687168748221942798475910261277585231089206502667816546785788734428,1730012621060119013316272117060130212413,1380067339047144425076529243235427669785561426931311250902916681307956,2803858665497933543661573348716832763278064685314491547922700863988457][3,3584217634602882250414174591984384801148279487761588373233371085330221,2850437460952442263625095594787096868582929687103122256419966625969481,120828043284204930985907233630576833469637665909287136193075466903137]" >> "$OUT"
add ""

add "##### RSA derive+d decrypt #####"
./sabaton.exe -decypher -rsa -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "derive:[n,e1,d1,e2]:[10911792316465922985916807702379449750396409, 37, 9142312481363340880084193654225213067375433,101]" "0" >> "$OUT"
add ""

add "##### RABIN #####"
add ""

add "##### ELLIPTIC CURVE MV decrypt #####"
./sabaton.exe -decypher -ellipticCurve -frag "mode:mv|q:3001|a:-7|b:2|P:[21,2819]|n:509|r:250" "[[[2447,854],38,105],[[1175,786],408,393]]" >> "$OUT"
add ""

###############################################################################
# ADDITIONAL VARIANTS FROM AAAA (best-effort automation)
###############################################################################
add "##### MERKLE #####"
./sabaton.exe -decypher -merkle -alph "abcdefghijklmnopqrstuvwxyz " -frag "key:89768834,19262404,21157368,65303715,53464120,101836023,67836715,80808166|p:120479184|w1:911578" "[242528773,121227938,174692058,105723487,21157368,223063961,105723487,288367676,254368368,223063961,254368368,21157368,195719915,121227938,105723487,121227938,254368368,21157368,142255795,173560202,223063961,108256487,223063961,161720607,344364796]" >> "$OUT"
add ""

add "##### GRAHAM #####"
./sabaton.exe -decypher -graham -alph "abcdefghijklmnopqrstuvwxyz " -frag "key:204768901270,89633072037,42563477532,246070679338,56786136228,7553306604,144151498682,128290894977|p:266857347661|w1:37021810743" "[579204863817,317273580774,268040751150,324826887378,468978386060,650709622566,42563477532,650709622566,514111430488,385820535511,268040751150,324826887378,468978386060,650709622566]" >> "$OUT"
add ""

add "##### RSA 36.1 decrypt with [n,e,d] #####"
./sabaton.exe -decypher -rsa -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "[n,e,d]:[1064981146987376853910813688626037,47,1019662800307062870687094878670643]" "845714650420490534999211092671223" >> "$OUT"
add ""

add "##### RSA 36.3 common modulus attack #####"
./sabaton.exe -decypher -rsa -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "mod:[n,e,c]:[34561700240436418904946007660686524210434811,59,27803039462374400827259181872367232196474352]|[n,e,c]:[34561700240436418904946007660686524210434811,95,9970122828808728753980666004212037946958957]" "0" >> "$OUT"
add ""

add "##### RABIN 66.1 #####"
./sabaton.exe -decypher -rabin -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "[p,q]:[76646242389071294489171,76646242389071294489203]" "2764847804273332122157312381958699759265286664" >> "$OUT"
add ""

add "##### BLUM-GOLDWASSER 66.2 #####"
./sabaton.exe -decypher -rabin -alph "ABCDEFGHIJKLMNOPQRSTUVWXYZ " -frag "bg:91127|91139|h:9" "[460,205,72,264,214,389,428,318,62,403,496,14,56,325,346,117,440,385,480,196,58,159,168,73,371,439,85,228,240,338119014]" >> "$OUT"
add ""

add "##### ELGAMAL signatures/dec (18 variant) #####"
./sabaton.exe -decypher -elgamal -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "1;2;3;4" "[1,3584217634602882250414174591984384801148279487761588373233371085330221,2,3572561055637859460099057608692160179234251257895318983525239092556191,1627211721333001232207181324,1380067339047144425076529243235427669785561426931311250902916681307956,679687168748221942798475910261277585231089206502667816546785788734428,1730012621060119013316272117060130212413,1380067339047144425076529243235427669785561426931311250902916681307956,2803858665497933543661573348716832763278064685314491547922700863988457][3,3584217634602882250414174591984384801148279487761588373233371085330221,2850437460952442263625095594787096868582929687103122256419966625969481,120828043284204930985907233630576833469637665909287136193075466903137]" >> "$OUT"
add ""

add "##### ELLIPTIC MV decrypt (67 variant) #####"
./sabaton.exe -decypher -ellipticCurve -frag "mode:mv|q:3001|a:-7|b:2|P:[21,2819]|n:509|r:371" "[[[2197,1940],83,276],[[2165,1485],223,215],[[2385,940],387,74],[[2220,209],200,440],[[2316,298],125,169],[[773,2357],132,22],[[377,880],336,55],[[2918,2037],299,78]]" >> "$OUT"
add ""

add "##### ELGAMAL EC signature (67 variant) #####"
add "# Sign m=100 with r=371 on same curve (manual via elliptic module not automated here)"
add ""

add "##### SHAMIR 29.1 #####"
./sabaton.exe -decypher -shamir -alph "ABCDEFGHIJKLMNOPQRSTUVWXYZ " -frag "1,2,3|152482782390306,12023267045466,639316297025722|695455899994403" "0" >> "$OUT"
add ""

add "##### ASMUTH-BLOOM 29.2 #####"
./sabaton.exe -decypher -asmuth -alph "ABCDEFGHIJKLMNOPQRSTUVWXYZ " -frag "19753545014,104598376909,45864401815|56797909229,80273229779,108483925343,134269474327|56797909229" "0" >> "$OUT"
add ""

add "##### ZKP tasks (7 variant) #####"
./sabaton.exe -decypher -zkp -frag "63723,100003,2,10842,100003" "0" >> "$OUT"
add ""
echo "Results written to $OUT"
