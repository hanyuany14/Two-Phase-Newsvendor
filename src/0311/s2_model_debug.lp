\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14] + profits_vars[15] + profits_vars[16]
   + profits_vars[17] + profits_vars[18] + profits_vars[19]
   + profits_vars[20] + profits_vars[21] + profits_vars[22]
   + profits_vars[23] + profits_vars[24] + profits_vars[25]
   + profits_vars[26] + profits_vars[27] + profits_vars[28]
   + profits_vars[29] + profits_vars[30] + profits_vars[31]
   + profits_vars[32] + profits_vars[33] + profits_vars[34]
   + profits_vars[35] + profits_vars[36] + profits_vars[37]
   + profits_vars[38] + profits_vars[39] + profits_vars[40]
   + profits_vars[41] + profits_vars[42] + profits_vars[43]
   + profits_vars[44] + profits_vars[45] + profits_vars[46]
   + profits_vars[47] + profits_vars[48] + profits_vars[49]
   + profits_vars[50] + profits_vars[51] + profits_vars[52]
   + profits_vars[53] + profits_vars[54] + profits_vars[55]
   + profits_vars[56] + profits_vars[57] + profits_vars[58]
   + profits_vars[59] + profits_vars[60] + profits_vars[61]
   + profits_vars[62] + profits_vars[63] + profits_vars[64]
   + profits_vars[65] + profits_vars[66] + profits_vars[67]
   + profits_vars[68] + profits_vars[69] + profits_vars[70]
   + profits_vars[71] + profits_vars[72] + profits_vars[73]
   + profits_vars[74] + profits_vars[75] + profits_vars[76]
   + profits_vars[77] + profits_vars[78] + profits_vars[79]
   + profits_vars[80] + profits_vars[81] + profits_vars[82]
   + profits_vars[83] + profits_vars[84] + profits_vars[85]
   + profits_vars[86] + profits_vars[87] + profits_vars[88]
   + profits_vars[89] + profits_vars[90] + profits_vars[91]
   + profits_vars[92] + profits_vars[93] + profits_vars[94]
   + profits_vars[95] + profits_vars[96] + profits_vars[97]
   + profits_vars[98] + profits_vars[99] + profits_vars[100]
   + profits_vars[101] + profits_vars[102] + profits_vars[103]
   + profits_vars[104] + profits_vars[105] + profits_vars[106]
   + profits_vars[107] + profits_vars[108] + profits_vars[109]
   + profits_vars[110] + profits_vars[111] + profits_vars[112]
   + profits_vars[113] + profits_vars[114] + profits_vars[115]
   + profits_vars[116] + profits_vars[117] + profits_vars[118]
   + profits_vars[119] + profits_vars[120] + profits_vars[121]
   + profits_vars[122] + profits_vars[123] + profits_vars[124]
   + profits_vars[125] + profits_vars[126] + profits_vars[127]
   + profits_vars[128] + profits_vars[129] + profits_vars[130]
   + profits_vars[131] + profits_vars[132] + profits_vars[133]
   + profits_vars[134] + profits_vars[135] + profits_vars[136]
   + profits_vars[137] + profits_vars[138] + profits_vars[139]
   + profits_vars[140] + profits_vars[141] + profits_vars[142]
   + profits_vars[143] + profits_vars[144] + profits_vars[145]
   + profits_vars[146] + profits_vars[147] + profits_vars[148]
   + profits_vars[149]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: - 0.1377749876203837 alphas[0] - 1.463947123977916 alphas[1]
   + 1.032261431224093 alphas[2] - 1.083559159598017 alphas[3] - alphas[4]
   + f_var[0] = 0
 R3: - 1857.770397056156 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 730.3707927279808
 R5: Q0_var[0] + Q_hat_adjusted_0 = 899.1427041512603
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 168.7086446709725
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.7326601786430774 alphas[0] - 1.014613274984077 alphas[1]
   + 0.498648104244913 alphas[2] + 0.481288343661414 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 1857.770397056156 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 869.6845049936463
 R15: Q0_var[1] + Q_hat_adjusted_1 = 1124.373517536452
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 254.633373824361
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.3306754452021703 alphas[0] + 0.4804815487491251 alphas[1]
   - 0.7276273688600782 alphas[2] + 0.9437855273712464 alphas[3]
   - alphas[4] + f_var[2] = 0
 R23: - 1857.770397056156 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1329.298118406662
 R25: Q0_var[2] + Q_hat_adjusted_2 = 1708.226251413412
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 378.8778909639552
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.1237219291831992 alphas[0] + 1.335159390007656 alphas[1]
   + 1.420129191828129 alphas[2] + 1.458149706239022 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 1857.770397056156 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 377.7611703831191
 R35: Q0_var[3] + Q_hat_adjusted_3 = 421.6317812682136
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 43.82977226896783
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.3097362005593667 alphas[0] + 0.2292226983763437 alphas[1]
   - 0.4732293109554516 alphas[2] - 1.30405062622969 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 1857.770397056156 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 1794.210237585196
 R45: Q0_var[4] + Q_hat_adjusted_4 = 2111.729426736025
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 317.4725421336088
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.4848915456578158 alphas[0] + 0.8226697598998868 alphas[1]
   - 0.0221891187456756 alphas[2] - 1.447868141815947 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 1857.770397056156 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1030.824017135466
 R55: Q0_var[5] + Q_hat_adjusted_5 = 1269.732059381323
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 238.8590514752216
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.259920167110256 alphas[0] + 1.160909155069043 alphas[1]
   + 0.6517339633154636 alphas[2] + 1.2527140264065 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 1857.770397056156 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 803.8809012446486
 R65: Q0_var[6] + Q_hat_adjusted_6 = 957.5982297912471
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 153.6709822174253
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.364043767244689 alphas[0] + 1.429002349390646 alphas[1]
   + 0.8103834737230072 alphas[2] + 0.527688532525546 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 1857.770397056156 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 403.1630961049649
 R75: Q0_var[7] + Q_hat_adjusted_7 = 534.2048546747362
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 130.9836691493348
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.621089187262248 alphas[0] - 0.8503097501729185 alphas[1]
   - 0.1504899505135745 alphas[2] + 1.065831069856636 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 1857.770397056156 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1266.450319396651
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1625.438318556864
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 358.9331194955274
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.4535207899110536 alphas[0] + 1.571422984738842 alphas[1]
   + 0.3596515255823725 alphas[2] + 1.266993471110587 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 1857.770397056156 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 752.9232712374273
 R95: Q0_var[9] + Q_hat_adjusted_9 = 932.0791287517279
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 179.1106038049684
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: - 1.006317288612526 alphas[0] - 1.002858123128412 alphas[1]
   + 0.0513351086078755 alphas[2] - 1.384686244121299 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 1857.770397056156 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 1652.066324940119
 R105: Q0_var[10] + Q_hat_adjusted_10 = 1951.607619071293
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 299.4777637564143
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: - 0.0665550968419857 alphas[0] + 1.111878796600655 alphas[1]
   + 0.4656234333574 alphas[2] + 1.466625713426974 alphas[3] - alphas[4]
   + f_var[11] = 0
 R113: - 1857.770397056156 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 1065.655092867825
 R115: Q0_var[11] + Q_hat_adjusted_11 = 1254.031533938886
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 188.3267912559151
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: - 0.2065365913049394 alphas[0] + 1.340514805390475 alphas[1]
   - 1.737381765985982 alphas[2] - 1.657370181869511 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 1857.770397056156 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 1980.867431116381
 R125: Q0_var[12] + Q_hat_adjusted_12 = 2442.306421828597
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 461.3656491148446
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 1.484981869478146 alphas[0] + 1.306330400563805 alphas[1]
   + 1.175710450631452 alphas[2] + 1.332772699279525 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 1857.770397056156 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 539.8468829401479
 R135: Q0_var[13] + Q_hat_adjusted_13 = 633.5992207510693
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 93.71170758669381
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: 1.570542089701374 alphas[0] - 0.668733918854781 alphas[1]
   + 1.303208191614615 alphas[2] - 1.261580773097114 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 1857.770397056156 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 449.3767629861672
 R145: Q0_var[14] + Q_hat_adjusted_14 = 529.8346672783546
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 80.42412628960201
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 1.512999902358935 alphas[0] + 0.7781356605649468 alphas[1]
   + 0.4450404925911721 alphas[2] - 0.2571334266325978 alphas[3]
   - alphas[4] + f_var[15] = 0
 R153: - 1857.770397056156 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 737.3331366144699
 R155: Q0_var[15] + Q_hat_adjusted_15 = 902.4136973175752
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 165.0386537877427
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 1.752243169309988 alphas[0] + 0.1841283139782117 alphas[1]
   - 0.416772325554038 alphas[2] + 0.4145690267442853 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 1857.770397056156 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1132.129505673431
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1447.332814356096
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 315.1440193232037
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 1.152538711559883 alphas[0] - 0.2798585383866669 alphas[1]
   - 0.6539445064201262 alphas[2] + 0.5010738073566936 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 1857.770397056156 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1488.526064406169
 R175: Q0_var[17] + Q_hat_adjusted_17 = 1887.758779069099
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 399.1573861422436
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.9578032015073822 alphas[0] - 1.307946923074003 alphas[1]
   + 0.2575810016909161 alphas[2] - 0.9026881000565862 alphas[3]
   - alphas[4] + f_var[18] = 0
 R183: - 1857.770397056156 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 1288.670206187575
 R185: Q0_var[18] + Q_hat_adjusted_18 = 1574.528869573287
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 285.8000152534103
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.286236754093806 alphas[0] - 0.8552263412864284 alphas[1]
   - 1.837556061802424 alphas[2] + 0.5949994327555658 alphas[3] - alphas[4]
   + f_var[19] = 0
 R193: - 1857.770397056156 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 2345.395139225224
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2937.636896602888
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 592.1756591175944
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 1.674563615370919 alphas[0] + 0.693494195622574 alphas[1]
   + 0.4154546257851501 alphas[2] - 0.564296985550078 alphas[3] - alphas[4]
   + f_var[20] = 0
 R203: - 1857.770397056156 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 701.7271029196665
 R205: Q0_var[20] + Q_hat_adjusted_20 = 906.0969644706238
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 204.3116202512401
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: - 1.032896231911416 alphas[0] + 1.164316541947729 alphas[1]
   - 0.8705162487653093 alphas[2] - 0.0907878295794809 alphas[3]
   - alphas[4] + f_var[21] = 0
 R213: - 1857.770397056156 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1393.512650710201
 R215: Q0_var[21] + Q_hat_adjusted_21 = 1764.413970128618
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 370.8283656243854
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.1744925900753842 alphas[0] + 1.422399622026228 alphas[1]
   - 0.5787944623871635 alphas[2] + 0.0186675154857767 alphas[3]
   - alphas[4] + f_var[22] = 0
 R223: - 1857.770397056156 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 1432.682604551746
 R225: Q0_var[22] + Q_hat_adjusted_22 = 1740.309598386797
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 307.5706109011828
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 0.9662859265782689 alphas[0] + 0.5874554698594764 alphas[1]
   - 1.189464187309562 alphas[2] - 1.389067874517465 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 1857.770397056156 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 1891.806475401024
 R235: Q0_var[23] + Q_hat_adjusted_23 = 2312.832578272942
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 420.9672444883834
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 1.401638950973156 alphas[0] + 0.7213333949652027 alphas[1]
   - 1.757330002963899 alphas[2] - 0.2170148371471542 alphas[3] - alphas[4]
   + f_var[24] = 0
 R243: - 1857.770397056156 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 2031.702061734177
 R245: Q0_var[24] + Q_hat_adjusted_24 = 2516.954392610623
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 485.1859808270665
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: - 0.4635344583495397 alphas[0] + 0.063405344478933 alphas[1]
   - 1.456531812980546 alphas[2] - 1.137755016042524 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 1857.770397056156 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 1943.65065166081
 R255: Q0_var[25] + Q_hat_adjusted_25 = 2418.950962300537
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 475.2298694714289
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: 1.311968231202461 alphas[0] - 0.7068266990475179 alphas[1]
   - 1.020704107429502 alphas[2] - 0.7951601347569953 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 1857.770397056156 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 1916.176524625012
 R265: Q0_var[26] + Q_hat_adjusted_26 = 2340.298668669879
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 424.0597340783264
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.553175753042803 alphas[0] - 0.7486854511926444 alphas[1]
   - 0.7899217801301555 alphas[2] + 1.528590985089338 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 1857.770397056156 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 1690.394039235398
 R275: Q0_var[27] + Q_hat_adjusted_27 = 2142.403751234692
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 451.9175557268797
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: - 0.0413596343670197 alphas[0] + 0.6487638472881786 alphas[1]
   + 0.472511430459086 alphas[2] - 0.9559467708198299 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 1857.770397056156 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 897.4119354793039
 R285: Q0_var[28] + Q_hat_adjusted_28 = 1072.246951048255
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 174.794673699044
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.3418906175083228 alphas[0] + 0.321844926206539 alphas[1]
   + 1.12613142061625 alphas[2] + 0.9254683954004678 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 1857.770397056156 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 262.1780804630767
 R295: Q0_var[29] + Q_hat_adjusted_29 = 372.6577870940253
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 110.4194879087977
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 0.8786032765791966 alphas[0] + 0.9958923067484952 alphas[1]
   + 1.422118510914746 alphas[2] - 1.395790277073855 alphas[3] - alphas[4]
   + f_var[30] = 0
 R303: - 1857.770397056156 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 271.0327822046679
 R305: Q0_var[30] + Q_hat_adjusted_30 = 282.9558112521219
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 10.58615204986268
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 700 Sold_0[30] - 700 Sold_1[30] + 700 Lost_0[30] + 700 Lost_1[30]
   + 300 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: - 0.9437760994697187 alphas[0] - 1.063776455017245 alphas[1]
   + 0.79824316315906 alphas[2] + 1.513820313033081 alphas[3] - alphas[4]
   + f_var[31] = 0
 R313: - 1857.770397056156 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 1097.106798181648
 R315: Q0_var[31] + Q_hat_adjusted_31 = 1325.053980375674
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 227.9024900319866
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 700 Sold_0[31] - 700 Sold_1[31] + 700 Lost_0[31] + 700 Lost_1[31]
   + 300 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: 0.1935468026550832 alphas[0] + 1.417402130333848 alphas[1]
   + 0.1356889276722403 alphas[2] + 0.5324296402958957 alphas[3]
   - alphas[4] + f_var[32] = 0
 R323: - 1857.770397056156 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 875.9743933206253
 R325: Q0_var[32] + Q_hat_adjusted_32 = 1086.994707932721
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 210.9594381737755
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 700 Sold_0[32] - 700 Sold_1[32] + 700 Lost_0[32] + 700 Lost_1[32]
   + 300 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: - 0.2079288667067903 alphas[0] - 0.7533371978101684 alphas[1]
   - 0.176533917976954 alphas[2] + 1.321671430795227 alphas[3] - alphas[4]
   + f_var[33] = 0
 R333: - 1857.770397056156 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 1413.147746884571
 R335: Q0_var[33] + Q_hat_adjusted_33 = 1759.68692463161
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 346.4596423887937
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 700 Sold_0[33] - 700 Sold_1[33] + 700 Lost_0[33] + 700 Lost_1[33]
   + 300 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: 1.757351189121949 alphas[0] - 1.030374810306114 alphas[1]
   - 1.050360664501698 alphas[2] + 0.0503825097146814 alphas[3] - alphas[4]
   + f_var[34] = 0
 R343: - 1857.770397056156 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 1964.532943950368
 R345: Q0_var[34] + Q_hat_adjusted_34 = 2409.38930880327
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 444.7870483758768
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 700 Sold_0[34] - 700 Sold_1[34] + 700 Lost_0[34] + 700 Lost_1[34]
   + 300 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: - 0.3838514553186348 alphas[0] - 1.02600777795809 alphas[1]
   - 1.696480111287648 alphas[2] - 1.114622344128488 alphas[3] - alphas[4]
   + f_var[35] = 0
 R353: - 1857.770397056156 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 2052.016358371682
 R355: Q0_var[35] + Q_hat_adjusted_35 = 2599.738671503496
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 547.63768789984
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 700 Sold_0[35] - 700 Sold_1[35] + 700 Lost_0[35] + 700 Lost_1[35]
   + 300 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 0.3640437159263562 alphas[0] + 0.7304740960075788 alphas[1]
   + 1.23180716250454 alphas[2] + 0.6504229158106599 alphas[3] - alphas[4]
   + f_var[36] = 0
 R363: - 1857.770397056156 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 311.3260626884876
 R365: Q0_var[36] + Q_hat_adjusted_36 = 396.8031162822365
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 85.42107893767221
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 700 Sold_0[36] - 700 Sold_1[36] + 700 Lost_0[36] + 700 Lost_1[36]
   + 300 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.3813432063754716 alphas[0] + 0.3425768209380515 alphas[1]
   + 1.267709793533505 alphas[2] + 1.148486906790564 alphas[3] - alphas[4]
   + f_var[37] = 0
 R373: - 1857.770397056156 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 397.5014433502334
 R375: Q0_var[37] + Q_hat_adjusted_37 = 497.6331291554777
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 100.0734071328117
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 700 Sold_0[37] - 700 Sold_1[37] + 700 Lost_0[37] + 700 Lost_1[37]
   + 300 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: - 1.549883248951645 alphas[0] - 0.3823710234057623 alphas[1]
   - 0.4121361184130283 alphas[2] + 0.5419172160108301 alphas[3]
   - alphas[4] + f_var[38] = 0
 R383: - 1857.770397056156 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 1881.617660557855
 R385: Q0_var[38] + Q_hat_adjusted_38 = 2254.629372270075
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 372.9723559444992
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 700 Sold_0[38] - 700 Sold_1[38] + 700 Lost_0[38] + 700 Lost_1[38]
   + 300 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 0.6133474149717183 alphas[0] + 0.6850370764557843 alphas[1]
   - 0.9540104512712915 alphas[2] - 1.093846763270934 alphas[3] - alphas[4]
   + f_var[39] = 0
 R393: - 1857.770397056156 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 1618.412973186389
 R395: Q0_var[39] + Q_hat_adjusted_39 = 2001.708260433537
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 383.2290680533446
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 700 Sold_0[39] - 700 Sold_1[39] + 700 Lost_0[39] + 700 Lost_1[39]
   + 300 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: 0.5390966352077166 alphas[0] + 0.3526152995005263 alphas[1]
   - 1.311606078764719 alphas[2] + 1.18417420843459 alphas[3] - alphas[4]
   + f_var[40] = 0
 R403: - 1857.770397056156 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 1863.590338352213
 R405: Q0_var[40] + Q_hat_adjusted_40 = 2324.083784362868
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 460.4162802920647
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 700 Sold_0[40] - 700 Sold_1[40] + 700 Lost_0[40] + 700 Lost_1[40]
   + 300 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: 0.2619055178534352 alphas[0] + 0.9419643151546333 alphas[1]
   - 1.616753413975551 alphas[2] + 0.889979776616903 alphas[3] - alphas[4]
   + f_var[41] = 0
 R413: - 1857.770397056156 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 1938.474854887897
 R415: Q0_var[41] + Q_hat_adjusted_41 = 2421.549881290937
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 483.0231304209928
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 700 Sold_0[41] - 700 Sold_1[41] + 700 Lost_0[41] + 700 Lost_1[41]
   + 300 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.6698800648346713 alphas[0] + 0.0508067423574906 alphas[1]
   - 1.782389250653439 alphas[2] + 1.426122692688451 alphas[3] - alphas[4]
   + f_var[42] = 0
 R423: - 1857.770397056156 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 2653.538574913986
 R425: Q0_var[42] + Q_hat_adjusted_42 = 3208.716629972328
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 555.1207929398457
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 700 Sold_0[42] - 700 Sold_1[42] + 700 Lost_0[42] + 700 Lost_1[42]
   + 300 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: 1.609195880951059 alphas[0] + 1.458893519890769 alphas[1]
   + 0.2487241569513961 alphas[2] - 0.7279742350994214 alphas[3]
   - alphas[4] + f_var[43] = 0
 R433: - 1857.770397056156 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 695.0864625406235
 R435: Q0_var[43] + Q_hat_adjusted_43 = 857.6579818500288
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 162.5087358697913
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 700 Sold_0[43] - 700 Sold_1[43] + 700 Lost_0[43] + 700 Lost_1[43]
   + 300 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: - 0.5595225760903506 alphas[0] - 1.101926353281677 alphas[1]
   + 0.3164709417828262 alphas[2] + 0.3221767224495976 alphas[3]
   - alphas[4] + f_var[44] = 0
 R443: - 1857.770397056156 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 1240.946505298687
 R445: Q0_var[44] + Q_hat_adjusted_44 = 1521.081716998501
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 280.0782332900822
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 700 Sold_0[44] - 700 Sold_1[44] + 700 Lost_0[44] + 700 Lost_1[44]
   + 300 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: - 0.5733640808347127 alphas[0] + 1.349214849693955 alphas[1]
   + 1.125703016782609 alphas[2] + 0.6076269376796377 alphas[3] - alphas[4]
   + f_var[45] = 0
 R453: - 1857.770397056156 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 365.4752556834671
 R455: Q0_var[45] + Q_hat_adjusted_45 = 446.1230699382257
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 80.60482418747404
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 700 Sold_0[45] - 700 Sold_1[45] + 700 Lost_0[45] + 700 Lost_1[45]
   + 300 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: 1.072301721396373 alphas[0] - 0.1491913474410184 alphas[1]
   - 0.7435866432784681 alphas[2] - 0.7716622448615245 alphas[3]
   - alphas[4] + f_var[46] = 0
 R463: - 1857.770397056156 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 1673.535330517504
 R465: Q0_var[46] + Q_hat_adjusted_46 = 2041.067922861127
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 367.4714805323185
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 700 Sold_0[46] - 700 Sold_1[46] + 700 Lost_0[46] + 700 Lost_1[46]
   + 300 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: 1.363552652341903 alphas[0] + 0.5697834446728656 alphas[1]
   - 0.6455415861558116 alphas[2] + 0.5226021045086005 alphas[3]
   - alphas[4] + f_var[47] = 0
 R473: - 1857.770397056156 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 1406.264080400039
 R475: Q0_var[47] + Q_hat_adjusted_47 = 1746.302917677532
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 339.9899378798741
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 700 Sold_0[47] - 700 Sold_1[47] + 700 Lost_0[47] + 700 Lost_1[47]
   + 300 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: 0.6967052626139382 alphas[0] - 0.3482689019206935 alphas[1]
   - 1.360022971848407 alphas[2] - 0.8025937266727698 alphas[3] - alphas[4]
   + f_var[48] = 0
 R483: - 1857.770397056156 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 2012.645614215695
 R485: Q0_var[48] + Q_hat_adjusted_48 = 2479.61961545245
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 466.9165313952075
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 700 Sold_0[48] - 700 Sold_1[48] + 700 Lost_0[48] + 700 Lost_1[48]
   + 300 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: 0.5240690626652507 alphas[0] - 1.643255348683263 alphas[1]
   + 1.301628049038135 alphas[2] - 1.098308981422492 alphas[3] - alphas[4]
   + f_var[49] = 0
 R493: - 1857.770397056156 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 518.0576627311875
 R495: Q0_var[49] + Q_hat_adjusted_49 = 646.8769413675645
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 128.7752493999675
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 700 Sold_0[49] - 700 Sold_1[49] + 700 Lost_0[49] + 700 Lost_1[49]
   + 300 Left_1[49] + profits_vars[49] = 0
 Fraction_lower_bound_50: Fraction[50] >= 0
 Fraction_upper_bound_50: Fraction[50] <= 1
 R502: - 0.2142319230434581 alphas[0] - 0.5789487715607019 alphas[1]
   - 0.0922463762577998 alphas[2] + 0.9227875423326392 alphas[3]
   - alphas[4] + f_var[50] = 0
 R503: - 1857.770397056156 Fraction[50] + Q0_var[50] = 0
 R504: Q0_var[50] + Lost_0_expr_50 = 1155.543890860777
 R505: Q0_var[50] + Q_hat_adjusted_50 = 1479.146126629022
 R506: Sold_0[50] - Q0_var[50] - Q1_var[50] + total_stock_second_period_50
   = 0
 R507: total_stock_second_period_50 + Lost_1_expr_50 = 323.5320368272452
 R508: Sold_1[50] + Left_1[50] - total_stock_second_period_50 = 0
 R509: - 700 Sold_0[50] - 700 Sold_1[50] + 700 Lost_0[50] + 700 Lost_1[50]
   + 300 Left_1[50] + profits_vars[50] = 0
 Fraction_lower_bound_51: Fraction[51] >= 0
 Fraction_upper_bound_51: Fraction[51] <= 1
 R512: 0.2562934790263433 alphas[0] + 1.490259719999384 alphas[1]
   - 0.3821621222339048 alphas[2] - 1.63481995791431 alphas[3] - alphas[4]
   + f_var[51] = 0
 R513: - 1857.770397056156 Fraction[51] + Q0_var[51] = 0
 R514: Q0_var[51] + Lost_0_expr_51 = 1203.586068617233
 R515: Q0_var[51] + Q_hat_adjusted_51 = 1460.086044058302
 R516: Sold_0[51] - Q0_var[51] - Q1_var[51] + total_stock_second_period_51
   = 0
 R517: total_stock_second_period_51 + Lost_1_expr_51 = 256.4495662927921
 R518: Sold_1[51] + Left_1[51] - total_stock_second_period_51 = 0
 R519: - 700 Sold_0[51] - 700 Sold_1[51] + 700 Lost_0[51] + 700 Lost_1[51]
   + 300 Left_1[51] + profits_vars[51] = 0
 Fraction_lower_bound_52: Fraction[52] >= 0
 Fraction_upper_bound_52: Fraction[52] <= 1
 R522: - 1.70944487706933 alphas[0] + 0.1506830731329518 alphas[1]
   + 0.7994426534814517 alphas[2] + 1.10984979281002 alphas[3] - alphas[4]
   + f_var[52] = 0
 R523: - 1857.770397056156 Fraction[52] + Q0_var[52] = 0
 R524: Q0_var[52] + Lost_0_expr_52 = 1128.752483923819
 R525: Q0_var[52] + Q_hat_adjusted_52 = 1317.337455260699
 R526: Sold_0[52] - Q0_var[52] - Q1_var[52] + total_stock_second_period_52
   = 0
 R527: total_stock_second_period_52 + Lost_1_expr_52 = 188.5412448682371
 R528: Sold_1[52] + Left_1[52] - total_stock_second_period_52 = 0
 R529: - 700 Sold_0[52] - 700 Sold_1[52] + 700 Lost_0[52] + 700 Lost_1[52]
   + 300 Left_1[52] + profits_vars[52] = 0
 Fraction_lower_bound_53: Fraction[53] >= 0
 Fraction_upper_bound_53: Fraction[53] <= 1
 R532: 1.459668748983132 alphas[0] - 0.1192885933854394 alphas[1]
   + 1.051865163985007 alphas[2] + 0.6752714218758934 alphas[3] - alphas[4]
   + f_var[53] = 0
 R533: - 1857.770397056156 Fraction[53] + Q0_var[53] = 0
 R534: Q0_var[53] + Lost_0_expr_53 = 606.111457300907
 R535: Q0_var[53] + Q_hat_adjusted_53 = 727.3936904579321
 R536: Sold_0[53] - Q0_var[53] - Q1_var[53] + total_stock_second_period_53
   = 0
 R537: total_stock_second_period_53 + Lost_1_expr_53 = 121.2366780078799
 R538: Sold_1[53] + Left_1[53] - total_stock_second_period_53 = 0
 R539: - 700 Sold_0[53] - 700 Sold_1[53] + 700 Lost_0[53] + 700 Lost_1[53]
   + 300 Left_1[53] + profits_vars[53] = 0
 Fraction_lower_bound_54: Fraction[54] >= 0
 Fraction_upper_bound_54: Fraction[54] <= 1
 R542: 1.07768580217236 alphas[0] - 0.2079990664346201 alphas[1]
   - 1.351386023907077 alphas[2] + 1.050870048305009 alphas[3] - alphas[4]
   + f_var[54] = 0
 R543: - 1857.770397056156 Fraction[54] + Q0_var[54] = 0
 R544: Q0_var[54] + Lost_0_expr_54 = 2153.158103040867
 R545: Q0_var[54] + Q_hat_adjusted_54 = 2631.595842020458
 R546: Sold_0[54] - Q0_var[54] - Q1_var[54] + total_stock_second_period_54
   = 0
 R547: total_stock_second_period_54 + Lost_1_expr_54 = 478.3687726861248
 R548: Sold_1[54] + Left_1[54] - total_stock_second_period_54 = 0
 R549: - 700 Sold_0[54] - 700 Sold_1[54] + 700 Lost_0[54] + 700 Lost_1[54]
   + 300 Left_1[54] + profits_vars[54] = 0
 Fraction_lower_bound_55: Fraction[55] >= 0
 Fraction_upper_bound_55: Fraction[55] <= 1
 R552: 1.247764833908228 alphas[0] - 0.7004217128163637 alphas[1]
   + 1.436390974688403 alphas[2] + 0.4903743888048382 alphas[3] - alphas[4]
   + f_var[55] = 0
 R553: - 1857.770397056156 Fraction[55] + Q0_var[55] = 0
 R554: Q0_var[55] + Lost_0_expr_55 = 255.1921841831005
 R555: Q0_var[55] + Q_hat_adjusted_55 = 342.0052245594126
 R556: Sold_0[55] - Q0_var[55] - Q1_var[55] + total_stock_second_period_55
   = 0
 R557: total_stock_second_period_55 + Lost_1_expr_55 = 86.74051335627254
 R558: Sold_1[55] + Left_1[55] - total_stock_second_period_55 = 0
 R559: - 700 Sold_0[55] - 700 Sold_1[55] + 700 Lost_0[55] + 700 Lost_1[55]
   + 300 Left_1[55] + profits_vars[55] = 0
 Fraction_lower_bound_56: Fraction[56] >= 0
 Fraction_upper_bound_56: Fraction[56] <= 1
 R562: - 0.5106863217619549 alphas[0] + 0.6688478285809211 alphas[1]
   + 0.002261169097532 alphas[2] + 0.0352572755155343 alphas[3] - alphas[4]
   + f_var[56] = 0
 R563: - 1857.770397056156 Fraction[56] + Q0_var[56] = 0
 R564: Q0_var[56] + Lost_0_expr_56 = 1120.695096417727
 R565: Q0_var[56] + Q_hat_adjusted_56 = 1379.377627710282
 R566: Sold_0[56] - Q0_var[56] - Q1_var[56] + total_stock_second_period_56
   = 0
 R567: total_stock_second_period_56 + Lost_1_expr_56 = 258.6176776940822
 R568: Sold_1[56] + Left_1[56] - total_stock_second_period_56 = 0
 R569: - 700 Sold_0[56] - 700 Sold_1[56] + 700 Lost_0[56] + 700 Lost_1[56]
   + 300 Left_1[56] + profits_vars[56] = 0
 Fraction_lower_bound_57: Fraction[57] >= 0
 Fraction_upper_bound_57: Fraction[57] <= 1
 R572: 0.9188783043577932 alphas[0] + 1.173208510257185 alphas[1]
   + 1.236980364346887 alphas[2] - 0.1071916538415056 alphas[3] - alphas[4]
   + f_var[57] = 0
 R573: - 1857.770397056156 Fraction[57] + Q0_var[57] = 0
 R574: Q0_var[57] + Lost_0_expr_57 = 445.5600231954294
 R575: Q0_var[57] + Q_hat_adjusted_57 = 491.3077435132751
 R576: Sold_0[57] - Q0_var[57] - Q1_var[57] + total_stock_second_period_57
   = 0
 R577: total_stock_second_period_57 + Lost_1_expr_57 = 45.69869434935595
 R578: Sold_1[57] + Left_1[57] - total_stock_second_period_57 = 0
 R579: - 700 Sold_0[57] - 700 Sold_1[57] + 700 Lost_0[57] + 700 Lost_1[57]
   + 300 Left_1[57] + profits_vars[57] = 0
 Fraction_lower_bound_58: Fraction[58] >= 0
 Fraction_upper_bound_58: Fraction[58] <= 1
 R582: 0.1572176405877692 alphas[0] + 0.2786141437696125 alphas[1]
   + 0.0494597484616135 alphas[2] - 1.228214910127852 alphas[3] - alphas[4]
   + f_var[58] = 0
 R583: - 1857.770397056156 Fraction[58] + Q0_var[58] = 0
 R584: Q0_var[58] + Lost_0_expr_58 = 1294.599640439763
 R585: Q0_var[58] + Q_hat_adjusted_58 = 1537.874850435709
 R586: Sold_0[58] - Q0_var[58] - Q1_var[58] + total_stock_second_period_58
   = 0
 R587: total_stock_second_period_58 + Lost_1_expr_58 = 243.2324475751845
 R588: Sold_1[58] + Left_1[58] - total_stock_second_period_58 = 0
 R589: - 700 Sold_0[58] - 700 Sold_1[58] + 700 Lost_0[58] + 700 Lost_1[58]
   + 300 Left_1[58] + profits_vars[58] = 0
 Fraction_lower_bound_59: Fraction[59] >= 0
 Fraction_upper_bound_59: Fraction[59] <= 1
 R592: 0.9505791670616883 alphas[0] - 1.6329887266194 alphas[1]
   - 1.77033422989719 alphas[2] - 1.349445773081045 alphas[3] - alphas[4]
   + f_var[59] = 0
 R593: - 1857.770397056156 Fraction[59] + Q0_var[59] = 0
 R594: Q0_var[59] + Lost_0_expr_59 = 2279.901858514709
 R595: Q0_var[59] + Q_hat_adjusted_59 = 2841.994995457991
 R596: Sold_0[59] - Q0_var[59] - Q1_var[59] + total_stock_second_period_59
   = 0
 R597: total_stock_second_period_59 + Lost_1_expr_59 = 562.0227537355461
 R598: Sold_1[59] + Left_1[59] - total_stock_second_period_59 = 0
 R599: - 700 Sold_0[59] - 700 Sold_1[59] + 700 Lost_0[59] + 700 Lost_1[59]
   + 300 Left_1[59] + profits_vars[59] = 0
 Fraction_lower_bound_60: Fraction[60] >= 0
 Fraction_upper_bound_60: Fraction[60] <= 1
 R602: 1.256131385441453 alphas[0] + 0.9756144247100236 alphas[1]
   + 0.1655318410568314 alphas[2] + 0.9145310907215168 alphas[3]
   - alphas[4] + f_var[60] = 0
 R603: - 1857.770397056156 Fraction[60] + Q0_var[60] = 0
 R604: Q0_var[60] + Lost_0_expr_60 = 1089.128129659249
 R605: Q0_var[60] + Q_hat_adjusted_60 = 1304.109005639912
 R606: Sold_0[60] - Q0_var[60] - Q1_var[60] + total_stock_second_period_60
   = 0
 R607: total_stock_second_period_60 + Lost_1_expr_60 = 214.9373091538704
 R608: Sold_1[60] + Left_1[60] - total_stock_second_period_60 = 0
 R609: - 700 Sold_0[60] - 700 Sold_1[60] + 700 Lost_0[60] + 700 Lost_1[60]
   + 300 Left_1[60] + profits_vars[60] = 0
 Fraction_lower_bound_61: Fraction[61] >= 0
 Fraction_upper_bound_61: Fraction[61] <= 1
 R612: 1.429883237262016 alphas[0] - 1.455227043357004 alphas[1]
   - 1.342839005389697 alphas[2] - 0.4478588964678674 alphas[3] - alphas[4]
   + f_var[61] = 0
 R613: - 1857.770397056156 Fraction[61] + Q0_var[61] = 0
 R614: Q0_var[61] + Lost_0_expr_61 = 2031.232422504364
 R615: Q0_var[61] + Q_hat_adjusted_61 = 2530.914824401066
 R616: Sold_0[61] - Q0_var[61] - Q1_var[61] + total_stock_second_period_61
   = 0
 R617: total_stock_second_period_61 + Lost_1_expr_61 = 499.6135742042624
 R618: Sold_1[61] + Left_1[61] - total_stock_second_period_61 = 0
 R619: - 700 Sold_0[61] - 700 Sold_1[61] + 700 Lost_0[61] + 700 Lost_1[61]
   + 300 Left_1[61] + profits_vars[61] = 0
 Fraction_lower_bound_62: Fraction[62] >= 0
 Fraction_upper_bound_62: Fraction[62] <= 1
 R622: - 0.5222041117931403 alphas[0] - 0.2338670812519379 alphas[1]
   + 1.231824271728301 alphas[2] + 1.281884106011574 alphas[3] - alphas[4]
   + f_var[62] = 0
 R623: - 1857.770397056156 Fraction[62] + Q0_var[62] = 0
 R624: Q0_var[62] + Lost_0_expr_62 = 640.8201850228692
 R625: Q0_var[62] + Q_hat_adjusted_62 = 769.6507979008475
 R626: Sold_0[62] - Q0_var[62] - Q1_var[62] + total_stock_second_period_62
   = 0
 R627: total_stock_second_period_62 + Lost_1_expr_62 = 128.7816338889592
 R628: Sold_1[62] + Left_1[62] - total_stock_second_period_62 = 0
 R629: - 700 Sold_0[62] - 700 Sold_1[62] + 700 Lost_0[62] + 700 Lost_1[62]
   + 300 Left_1[62] + profits_vars[62] = 0
 Fraction_lower_bound_63: Fraction[63] >= 0
 Fraction_upper_bound_63: Fraction[63] <= 1
 R632: 1.330455025559286 alphas[0] + 0.0607913835153343 alphas[1]
   + 0.6960988390008416 alphas[2] + 0.1065951752817167 alphas[3]
   - alphas[4] + f_var[63] = 0
 R633: - 1857.770397056156 Fraction[63] + Q0_var[63] = 0
 R634: Q0_var[63] + Lost_0_expr_63 = 607.7469194774691
 R635: Q0_var[63] + Q_hat_adjusted_63 = 768.4993927470873
 R636: Sold_0[63] - Q0_var[63] - Q1_var[63] + total_stock_second_period_63
   = 0
 R637: total_stock_second_period_63 + Lost_1_expr_63 = 160.6977291265339
 R638: Sold_1[63] + Left_1[63] - total_stock_second_period_63 = 0
 R639: - 700 Sold_0[63] - 700 Sold_1[63] + 700 Lost_0[63] + 700 Lost_1[63]
   + 300 Left_1[63] + profits_vars[63] = 0
 Fraction_lower_bound_64: Fraction[64] >= 0
 Fraction_upper_bound_64: Fraction[64] <= 1
 R642: 1.121645022352866 alphas[0] - 1.380820096790804 alphas[1]
   + 0.2348346247134404 alphas[2] + 0.5687513309656107 alphas[3]
   - alphas[4] + f_var[64] = 0
 R643: - 1857.770397056156 Fraction[64] + Q0_var[64] = 0
 R644: Q0_var[64] + Lost_0_expr_64 = 922.4950129198998
 R645: Q0_var[64] + Q_hat_adjusted_64 = 1208.344583204743
 R646: Sold_0[64] - Q0_var[64] - Q1_var[64] + total_stock_second_period_64
   = 0
 R647: total_stock_second_period_64 + Lost_1_expr_64 = 285.7850890080398
 R648: Sold_1[64] + Left_1[64] - total_stock_second_period_64 = 0
 R649: - 700 Sold_0[64] - 700 Sold_1[64] + 700 Lost_0[64] + 700 Lost_1[64]
   + 300 Left_1[64] + profits_vars[64] = 0
 Fraction_lower_bound_65: Fraction[65] >= 0
 Fraction_upper_bound_65: Fraction[65] <= 1
 R652: 0.5061398264739483 alphas[0] + 0.0550519971218631 alphas[1]
   + 0.248687796870595 alphas[2] + 0.5884800601904473 alphas[3] - alphas[4]
   + f_var[65] = 0
 R653: - 1857.770397056156 Fraction[65] + Q0_var[65] = 0
 R654: Q0_var[65] + Lost_0_expr_65 = 1114.227159963983
 R655: Q0_var[65] + Q_hat_adjusted_65 = 1354.394472161681
 R656: Sold_0[65] - Q0_var[65] - Q1_var[65] + total_stock_second_period_65
   = 0
 R657: total_stock_second_period_65 + Lost_1_expr_65 = 240.1013189266381
 R658: Sold_1[65] + Left_1[65] - total_stock_second_period_65 = 0
 R659: - 700 Sold_0[65] - 700 Sold_1[65] + 700 Lost_0[65] + 700 Lost_1[65]
   + 300 Left_1[65] + profits_vars[65] = 0
 Fraction_lower_bound_66: Fraction[66] >= 0
 Fraction_upper_bound_66: Fraction[66] <= 1
 R662: - 1.110967167466905 alphas[0] - 0.8454717313521594 alphas[1]
   - 0.6964269811850159 alphas[2] + 0.026722330226989 alphas[3] - alphas[4]
   + f_var[66] = 0
 R663: - 1857.770397056156 Fraction[66] + Q0_var[66] = 0
 R664: Q0_var[66] + Lost_0_expr_66 = 1721.491955556769
 R665: Q0_var[66] + Q_hat_adjusted_66 = 2140.763136634973
 R666: Sold_0[66] - Q0_var[66] - Q1_var[66] + total_stock_second_period_66
   = 0
 R667: total_stock_second_period_66 + Lost_1_expr_66 = 419.2020117553433
 R668: Sold_1[66] + Left_1[66] - total_stock_second_period_66 = 0
 R669: - 700 Sold_0[66] - 700 Sold_1[66] + 700 Lost_0[66] + 700 Lost_1[66]
   + 300 Left_1[66] + profits_vars[66] = 0
 Fraction_lower_bound_67: Fraction[67] >= 0
 Fraction_upper_bound_67: Fraction[67] <= 1
 R672: 1.477344604679948 alphas[0] + 0.2570825614334498 alphas[1]
   + 0.4404243983079117 alphas[2] - 0.8095491283351618 alphas[3]
   - alphas[4] + f_var[67] = 0
 R673: - 1857.770397056156 Fraction[67] + Q0_var[67] = 0
 R674: Q0_var[67] + Lost_0_expr_67 = 1183.261882704097
 R675: Q0_var[67] + Q_hat_adjusted_67 = 1361.31345749922
 R676: Sold_0[67] - Q0_var[67] - Q1_var[67] + total_stock_second_period_67
   = 0
 R677: total_stock_second_period_67 + Lost_1_expr_67 = 178.006844404126
 R678: Sold_1[67] + Left_1[67] - total_stock_second_period_67 = 0
 R679: - 700 Sold_0[67] - 700 Sold_1[67] + 700 Lost_0[67] + 700 Lost_1[67]
   + 300 Left_1[67] + profits_vars[67] = 0
 Fraction_lower_bound_68: Fraction[68] >= 0
 Fraction_upper_bound_68: Fraction[68] <= 1
 R682: - 1.171578736051981 alphas[0] - 1.455431987909636 alphas[1]
   - 0.8439284376522775 alphas[2] + 1.425004158439644 alphas[3] - alphas[4]
   + f_var[68] = 0
 R683: - 1857.770397056156 Fraction[68] + Q0_var[68] = 0
 R684: Q0_var[68] + Lost_0_expr_68 = 1815.729948644613
 R685: Q0_var[68] + Q_hat_adjusted_68 = 2294.988996326567
 R686: Sold_0[68] - Q0_var[68] - Q1_var[68] + total_stock_second_period_68
   = 0
 R687: total_stock_second_period_68 + Lost_1_expr_68 = 479.1930478028278
 R688: Sold_1[68] + Left_1[68] - total_stock_second_period_68 = 0
 R689: - 700 Sold_0[68] - 700 Sold_1[68] + 700 Lost_0[68] + 700 Lost_1[68]
   + 300 Left_1[68] + profits_vars[68] = 0
 Fraction_lower_bound_69: Fraction[69] >= 0
 Fraction_upper_bound_69: Fraction[69] <= 1
 R692: 1.480930408852247 alphas[0] - 0.7296944652902129 alphas[1]
   - 0.5843773695207689 alphas[2] - 1.314693254718296 alphas[3] - alphas[4]
   + f_var[69] = 0
 R693: - 1857.770397056156 Fraction[69] + Q0_var[69] = 0
 R694: Q0_var[69] + Lost_0_expr_69 = 1646.182380735384
 R695: Q0_var[69] + Q_hat_adjusted_69 = 2000.382070700075
 R696: Sold_0[69] - Q0_var[69] - Q1_var[69] + total_stock_second_period_69
   = 0
 R697: total_stock_second_period_69 + Lost_1_expr_69 = 354.147251345354
 R698: Sold_1[69] + Left_1[69] - total_stock_second_period_69 = 0
 R699: - 700 Sold_0[69] - 700 Sold_1[69] + 700 Lost_0[69] + 700 Lost_1[69]
   + 300 Left_1[69] + profits_vars[69] = 0
 Fraction_lower_bound_70: Fraction[70] >= 0
 Fraction_upper_bound_70: Fraction[70] <= 1
 R702: - 1.666844442239888 alphas[0] - 0.7622379104676235 alphas[1]
   + 0.2509533701302887 alphas[2] - 0.8264141525303725 alphas[3]
   - alphas[4] + f_var[70] = 0
 R703: - 1857.770397056156 Fraction[70] + Q0_var[70] = 0
 R704: Q0_var[70] + Lost_0_expr_70 = 1393.718420667341
 R705: Q0_var[70] + Q_hat_adjusted_70 = 1670.041505913331
 R706: Sold_0[70] - Q0_var[70] - Q1_var[70] + total_stock_second_period_70
   = 0
 R707: total_stock_second_period_70 + Lost_1_expr_70 = 276.27484325421
 R708: Sold_1[70] + Left_1[70] - total_stock_second_period_70 = 0
 R709: - 700 Sold_0[70] - 700 Sold_1[70] + 700 Lost_0[70] + 700 Lost_1[70]
   + 300 Left_1[70] + profits_vars[70] = 0
 Fraction_lower_bound_71: Fraction[71] >= 0
 Fraction_upper_bound_71: Fraction[71] <= 1
 R712: 0.1488493207595021 alphas[0] + 0.4988768225889902 alphas[1]
   + 0.1375033272527315 alphas[2] + 1.062018045765469 alphas[3] - alphas[4]
   + f_var[71] = 0
 R713: - 1857.770397056156 Fraction[71] + Q0_var[71] = 0
 R714: Q0_var[71] + Lost_0_expr_71 = 1100.421304225794
 R715: Q0_var[71] + Q_hat_adjusted_71 = 1350.430584731138
 R716: Sold_0[71] - Q0_var[71] - Q1_var[71] + total_stock_second_period_71
   = 0
 R717: total_stock_second_period_71 + Lost_1_expr_71 = 249.9663810258077
 R718: Sold_1[71] + Left_1[71] - total_stock_second_period_71 = 0
 R719: - 700 Sold_0[71] - 700 Sold_1[71] + 700 Lost_0[71] + 700 Lost_1[71]
   + 300 Left_1[71] + profits_vars[71] = 0
 Fraction_lower_bound_72: Fraction[72] >= 0
 Fraction_upper_bound_72: Fraction[72] <= 1
 R722: - 1.667922910946795 alphas[0] - 0.9560553224349836 alphas[1]
   - 0.498559131494877 alphas[2] - 1.517594100948176 alphas[3] - alphas[4]
   + f_var[72] = 0
 R723: - 1857.770397056156 Fraction[72] + Q0_var[72] = 0
 R724: Q0_var[72] + Lost_0_expr_72 = 1597.042907110813
 R725: Q0_var[72] + Q_hat_adjusted_72 = 1979.766791615007
 R726: Sold_0[72] - Q0_var[72] - Q1_var[72] + total_stock_second_period_72
   = 0
 R727: total_stock_second_period_72 + Lost_1_expr_72 = 382.6550906991539
 R728: Sold_1[72] + Left_1[72] - total_stock_second_period_72 = 0
 R729: - 700 Sold_0[72] - 700 Sold_1[72] + 700 Lost_0[72] + 700 Lost_1[72]
   + 300 Left_1[72] + profits_vars[72] = 0
 Fraction_lower_bound_73: Fraction[73] >= 0
 Fraction_upper_bound_73: Fraction[73] <= 1
 R732: - 0.3381202537139655 alphas[0] - 0.546703208058506 alphas[1]
   + 1.396343711197518 alphas[2] - 0.0535347559982192 alphas[3] - alphas[4]
   + f_var[73] = 0
 R733: - 1857.770397056156 Fraction[73] + Q0_var[73] = 0
 R734: Q0_var[73] + Lost_0_expr_73 = 525.3242745557617
 R735: Q0_var[73] + Q_hat_adjusted_73 = 623.6938175877717
 R736: Sold_0[73] - Q0_var[73] - Q1_var[73] + total_stock_second_period_73
   = 0
 R737: total_stock_second_period_73 + Lost_1_expr_73 = 98.36355005637778
 R738: Sold_1[73] + Left_1[73] - total_stock_second_period_73 = 0
 R739: - 700 Sold_0[73] - 700 Sold_1[73] + 700 Lost_0[73] + 700 Lost_1[73]
   + 300 Left_1[73] + profits_vars[73] = 0
 Fraction_lower_bound_74: Fraction[74] >= 0
 Fraction_upper_bound_74: Fraction[74] <= 1
 R742: - 0.8187387275632921 alphas[0] + 0.7962670679253425 alphas[1]
   - 1.222013794811172 alphas[2] - 1.72272764912815 alphas[3] - alphas[4]
   + f_var[74] = 0
 R743: - 1857.770397056156 Fraction[74] + Q0_var[74] = 0
 R744: Q0_var[74] + Lost_0_expr_74 = 1443.002268030613
 R745: Q0_var[74] + Q_hat_adjusted_74 = 1855.737874828348
 R746: Sold_0[74] - Q0_var[74] - Q1_var[74] + total_stock_second_period_74
   = 0
 R747: total_stock_second_period_74 + Lost_1_expr_74 = 412.657798536112
 R748: Sold_1[74] + Left_1[74] - total_stock_second_period_74 = 0
 R749: - 700 Sold_0[74] - 700 Sold_1[74] + 700 Lost_0[74] + 700 Lost_1[74]
   + 300 Left_1[74] + profits_vars[74] = 0
 Fraction_lower_bound_75: Fraction[75] >= 0
 Fraction_upper_bound_75: Fraction[75] <= 1
 R752: 1.684417152412663 alphas[0] + 1.065787861935759 alphas[1]
   - 0.6339150391866228 alphas[2] + 0.9917995188351328 alphas[3]
   - alphas[4] + f_var[75] = 0
 R753: - 1857.770397056156 Fraction[75] + Q0_var[75] = 0
 R754: Q0_var[75] + Lost_0_expr_75 = 1434.238648658598
 R755: Q0_var[75] + Q_hat_adjusted_75 = 1757.03403436736
 R756: Sold_0[75] - Q0_var[75] - Q1_var[75] + total_stock_second_period_75
   = 0
 R757: total_stock_second_period_75 + Lost_1_expr_75 = 322.7261684746407
 R758: Sold_1[75] + Left_1[75] - total_stock_second_period_75 = 0
 R759: - 700 Sold_0[75] - 700 Sold_1[75] + 700 Lost_0[75] + 700 Lost_1[75]
   + 300 Left_1[75] + profits_vars[75] = 0
 Fraction_lower_bound_76: Fraction[76] >= 0
 Fraction_upper_bound_76: Fraction[76] <= 1
 R762: 0.8133446630263352 alphas[0] - 1.090382033313018 alphas[1]
   - 0.8876936659287817 alphas[2] - 0.1502572480239133 alphas[3]
   - alphas[4] + f_var[76] = 0
 R763: - 1857.770397056156 Fraction[76] + Q0_var[76] = 0
 R764: Q0_var[76] + Lost_0_expr_76 = 1617.105693926821
 R765: Q0_var[76] + Q_hat_adjusted_76 = 2048.8986060276
 R766: Sold_0[76] - Q0_var[76] - Q1_var[76] + total_stock_second_period_76
   = 0
 R767: total_stock_second_period_76 + Lost_1_expr_76 = 431.727956220174
 R768: Sold_1[76] + Left_1[76] - total_stock_second_period_76 = 0
 R769: - 700 Sold_0[76] - 700 Sold_1[76] + 700 Lost_0[76] + 700 Lost_1[76]
   + 300 Left_1[76] + profits_vars[76] = 0
 Fraction_lower_bound_77: Fraction[77] >= 0
 Fraction_upper_bound_77: Fraction[77] <= 1
 R772: 1.394766260219685 alphas[0] - 1.642350843514383 alphas[1]
   - 0.2284814864237796 alphas[2] + 0.6772767143745821 alphas[3]
   - alphas[4] + f_var[77] = 0
 R773: - 1857.770397056156 Fraction[77] + Q0_var[77] = 0
 R774: Q0_var[77] + Lost_0_expr_77 = 1675.024757495721
 R775: Q0_var[77] + Q_hat_adjusted_77 = 2034.911377897422
 R776: Sold_0[77] - Q0_var[77] - Q1_var[77] + total_stock_second_period_77
   = 0
 R777: total_stock_second_period_77 + Lost_1_expr_77 = 359.8332301529997
 R778: Sold_1[77] + Left_1[77] - total_stock_second_period_77 = 0
 R779: - 700 Sold_0[77] - 700 Sold_1[77] + 700 Lost_0[77] + 700 Lost_1[77]
   + 300 Left_1[77] + profits_vars[77] = 0
 Fraction_lower_bound_78: Fraction[78] >= 0
 Fraction_upper_bound_78: Fraction[78] <= 1
 R782: 0.7656710167348737 alphas[0] + 0.056629263523689 alphas[1]
   + 1.255623260245099 alphas[2] + 0.6301753119154556 alphas[3] - alphas[4]
   + f_var[78] = 0
 R783: - 1857.770397056156 Fraction[78] + Q0_var[78] = 0
 R784: Q0_var[78] + Lost_0_expr_78 = 454.1989492184744
 R785: Q0_var[78] + Q_hat_adjusted_78 = 547.2227849595122
 R786: Sold_0[78] - Q0_var[78] - Q1_var[78] + total_stock_second_period_78
   = 0
 R787: total_stock_second_period_78 + Lost_1_expr_78 = 92.96767687391183
 R788: Sold_1[78] + Left_1[78] - total_stock_second_period_78 = 0
 R789: - 700 Sold_0[78] - 700 Sold_1[78] + 700 Lost_0[78] + 700 Lost_1[78]
   + 300 Left_1[78] + profits_vars[78] = 0
 Fraction_lower_bound_79: Fraction[79] >= 0
 Fraction_upper_bound_79: Fraction[79] <= 1
 R792: 1.400018178947205 alphas[0] - 0.3938479307469775 alphas[1]
   + 0.2305092150612923 alphas[2] - 0.3402022578856947 alphas[3]
   - alphas[4] + f_var[79] = 0
 R793: - 1857.770397056156 Fraction[79] + Q0_var[79] = 0
 R794: Q0_var[79] + Lost_0_expr_79 = 1237.572879628393
 R795: Q0_var[79] + Q_hat_adjusted_79 = 1475.235734004859
 R796: Sold_0[79] - Q0_var[79] - Q1_var[79] + total_stock_second_period_79
   = 0
 R797: total_stock_second_period_79 + Lost_1_expr_79 = 237.6161965135755
 R798: Sold_1[79] + Left_1[79] - total_stock_second_period_79 = 0
 R799: - 700 Sold_0[79] - 700 Sold_1[79] + 700 Lost_0[79] + 700 Lost_1[79]
   + 300 Left_1[79] + profits_vars[79] = 0
 Fraction_lower_bound_80: Fraction[80] >= 0
 Fraction_upper_bound_80: Fraction[80] <= 1
 R802: 0.6875703460464334 alphas[0] - 1.298355407763292 alphas[1]
   + 0.2293330231283329 alphas[2] - 1.458599138109167 alphas[3] - alphas[4]
   + f_var[80] = 0
 R803: - 1857.770397056156 Fraction[80] + Q0_var[80] = 0
 R804: Q0_var[80] + Lost_0_expr_80 = 1108.194004189659
 R805: Q0_var[80] + Q_hat_adjusted_80 = 1372.789744228043
 R806: Sold_0[80] - Q0_var[80] - Q1_var[80] + total_stock_second_period_80
   = 0
 R807: total_stock_second_period_80 + Lost_1_expr_80 = 264.5378697298788
 R808: Sold_1[80] + Left_1[80] - total_stock_second_period_80 = 0
 R809: - 700 Sold_0[80] - 700 Sold_1[80] + 700 Lost_0[80] + 700 Lost_1[80]
   + 300 Left_1[80] + profits_vars[80] = 0
 Fraction_lower_bound_81: Fraction[81] >= 0
 Fraction_upper_bound_81: Fraction[81] <= 1
 R812: 0.3433170669471007 alphas[0] + 0.0597332871573194 alphas[1]
   + 0.5228179317182622 alphas[2] - 1.064703196541337 alphas[3] - alphas[4]
   + f_var[81] = 0
 R813: - 1857.770397056156 Fraction[81] + Q0_var[81] = 0
 R814: Q0_var[81] + Lost_0_expr_81 = 528.4894221338071
 R815: Q0_var[81] + Q_hat_adjusted_81 = 711.366980843838
 R816: Sold_0[81] - Q0_var[81] - Q1_var[81] + total_stock_second_period_81
   = 0
 R817: total_stock_second_period_81 + Lost_1_expr_81 = 182.8220714829393
 R818: Sold_1[81] + Left_1[81] - total_stock_second_period_81 = 0
 R819: - 700 Sold_0[81] - 700 Sold_1[81] + 700 Lost_0[81] + 700 Lost_1[81]
   + 300 Left_1[81] + profits_vars[81] = 0
 Fraction_lower_bound_82: Fraction[82] >= 0
 Fraction_upper_bound_82: Fraction[82] <= 1
 R822: 1.59517248602961 alphas[0] - 1.617623317506734 alphas[1]
   + 1.535870269872266 alphas[2] - 0.7896396296472439 alphas[3] - alphas[4]
   + f_var[82] = 0
 R823: - 1857.770397056156 Fraction[82] + Q0_var[82] = 0
 R824: Q0_var[82] + Lost_0_expr_82 = 546.8272854362976
 R825: Q0_var[82] + Q_hat_adjusted_82 = 632.750197387192
 R826: Sold_0[82] - Q0_var[82] - Q1_var[82] + total_stock_second_period_82
   = 0
 R827: total_stock_second_period_82 + Lost_1_expr_82 = 85.88213386236083
 R828: Sold_1[82] + Left_1[82] - total_stock_second_period_82 = 0
 R829: - 700 Sold_0[82] - 700 Sold_1[82] + 700 Lost_0[82] + 700 Lost_1[82]
   + 300 Left_1[82] + profits_vars[82] = 0
 Fraction_lower_bound_83: Fraction[83] >= 0
 Fraction_upper_bound_83: Fraction[83] <= 1
 R832: - 0.6514335293928469 alphas[0] - 0.3421929710970733 alphas[1]
   - 0.9256685231517656 alphas[2] - 0.2323327072539335 alphas[3]
   - alphas[4] + f_var[83] = 0
 R833: - 1857.770397056156 Fraction[83] + Q0_var[83] = 0
 R834: Q0_var[83] + Lost_0_expr_83 = 1703.68743400237
 R835: Q0_var[83] + Q_hat_adjusted_83 = 2129.872551500837
 R836: Sold_0[83] - Q0_var[83] - Q1_var[83] + total_stock_second_period_83
   = 0
 R837: total_stock_second_period_83 + Lost_1_expr_83 = 426.1229907358042
 R838: Sold_1[83] + Left_1[83] - total_stock_second_period_83 = 0
 R839: - 700 Sold_0[83] - 700 Sold_1[83] + 700 Lost_0[83] + 700 Lost_1[83]
   + 300 Left_1[83] + profits_vars[83] = 0
 Fraction_lower_bound_84: Fraction[84] >= 0
 Fraction_upper_bound_84: Fraction[84] <= 1
 R842: - 0.2013766906815251 alphas[0] - 1.173039602344419 alphas[1]
   + 1.258033811088646 alphas[2] - 1.460559014357296 alphas[3] - alphas[4]
   + f_var[84] = 0
 R843: - 1857.770397056156 Fraction[84] + Q0_var[84] = 0
 R844: Q0_var[84] + Lost_0_expr_84 = 508.7821821660282
 R845: Q0_var[84] + Q_hat_adjusted_84 = 631.0719118775564
 R846: Sold_0[84] - Q0_var[84] - Q1_var[84] + total_stock_second_period_84
   = 0
 R847: total_stock_second_period_84 + Lost_1_expr_84 = 122.2301980049901
 R848: Sold_1[84] + Left_1[84] - total_stock_second_period_84 = 0
 R849: - 700 Sold_0[84] - 700 Sold_1[84] + 700 Lost_0[84] + 700 Lost_1[84]
   + 300 Left_1[84] + profits_vars[84] = 0
 Fraction_lower_bound_85: Fraction[85] >= 0
 Fraction_upper_bound_85: Fraction[85] <= 1
 R852: 0.875621701231364 alphas[0] - 1.471706452053066 alphas[1]
   - 0.4699536311598755 alphas[2] - 0.00634777082273264 alphas[3]
   - alphas[4] + f_var[85] = 0
 R853: - 1857.770397056156 Fraction[85] + Q0_var[85] = 0
 R854: Q0_var[85] + Lost_0_expr_85 = 1696.698661340503
 R855: Q0_var[85] + Q_hat_adjusted_85 = 2082.967486559034
 R856: Sold_0[85] - Q0_var[85] - Q1_var[85] + total_stock_second_period_85
   = 0
 R857: total_stock_second_period_85 + Lost_1_expr_85 = 386.2102326034618
 R858: Sold_1[85] + Left_1[85] - total_stock_second_period_85 = 0
 R859: - 700 Sold_0[85] - 700 Sold_1[85] + 700 Lost_0[85] + 700 Lost_1[85]
   + 300 Left_1[85] + profits_vars[85] = 0
 Fraction_lower_bound_86: Fraction[86] >= 0
 Fraction_upper_bound_86: Fraction[86] <= 1
 R862: - 0.0463644449049768 alphas[0] - 1.155259618659588 alphas[1]
   - 0.8072148162093365 alphas[2] - 1.296627416625219 alphas[3] - alphas[4]
   + f_var[86] = 0
 R863: - 1857.770397056156 Fraction[86] + Q0_var[86] = 0
 R864: Q0_var[86] + Lost_0_expr_86 = 1733.507241912456
 R865: Q0_var[86] + Q_hat_adjusted_86 = 2152.096566806757
 R866: Sold_0[86] - Q0_var[86] - Q1_var[86] + total_stock_second_period_86
   = 0
 R867: total_stock_second_period_86 + Lost_1_expr_86 = 418.5220379925635
 R868: Sold_1[86] + Left_1[86] - total_stock_second_period_86 = 0
 R869: - 700 Sold_0[86] - 700 Sold_1[86] + 700 Lost_0[86] + 700 Lost_1[86]
   + 300 Left_1[86] + profits_vars[86] = 0
 Fraction_lower_bound_87: Fraction[87] >= 0
 Fraction_upper_bound_87: Fraction[87] <= 1
 R872: 1.488646077644622 alphas[0] + 1.069600552026418 alphas[1]
   - 0.5690626102175452 alphas[2] - 1.161837709764985 alphas[3] - alphas[4]
   + f_var[87] = 0
 R873: - 1857.770397056156 Fraction[87] + Q0_var[87] = 0
 R874: Q0_var[87] + Lost_0_expr_87 = 1261.496370150473
 R875: Q0_var[87] + Q_hat_adjusted_87 = 1551.553849579981
 R876: Sold_0[87] - Q0_var[87] - Q1_var[87] + total_stock_second_period_87
   = 0
 R877: total_stock_second_period_87 + Lost_1_expr_87 = 289.9950482574426
 R878: Sold_1[87] + Left_1[87] - total_stock_second_period_87 = 0
 R879: - 700 Sold_0[87] - 700 Sold_1[87] + 700 Lost_0[87] + 700 Lost_1[87]
   + 300 Left_1[87] + profits_vars[87] = 0
 Fraction_lower_bound_88: Fraction[88] >= 0
 Fraction_upper_bound_88: Fraction[88] <= 1
 R882: - 0.2347903519342727 alphas[0] - 0.5224148485271782 alphas[1]
   - 1.697876804680736 alphas[2] + 0.9357335958625826 alphas[3] - alphas[4]
   + f_var[88] = 0
 R883: - 1857.770397056156 Fraction[88] + Q0_var[88] = 0
 R884: Q0_var[88] + Lost_0_expr_88 = 2335.536163489294
 R885: Q0_var[88] + Q_hat_adjusted_88 = 2888.215096360962
 R886: Sold_0[88] - Q0_var[88] - Q1_var[88] + total_stock_second_period_88
   = 0
 R887: total_stock_second_period_88 + Lost_1_expr_88 = 552.5971811280234
 R888: Sold_1[88] + Left_1[88] - total_stock_second_period_88 = 0
 R889: - 700 Sold_0[88] - 700 Sold_1[88] + 700 Lost_0[88] + 700 Lost_1[88]
   + 300 Left_1[88] + profits_vars[88] = 0
 Fraction_lower_bound_89: Fraction[89] >= 0
 Fraction_upper_bound_89: Fraction[89] <= 1
 R892: - 1.498209828142057 alphas[0] + 0.2590868503557352 alphas[1]
   + 1.280607256537937 alphas[2] - 0.9495723701991944 alphas[3] - alphas[4]
   + f_var[89] = 0
 R893: - 1857.770397056156 Fraction[89] + Q0_var[89] = 0
 R894: Q0_var[89] + Lost_0_expr_89 = 316.334619516777
 R895: Q0_var[89] + Q_hat_adjusted_89 = 405.251665215239
 R896: Sold_0[89] - Q0_var[89] - Q1_var[89] + total_stock_second_period_89
   = 0
 R897: total_stock_second_period_89 + Lost_1_expr_89 = 88.85781446285435
 R898: Sold_1[89] + Left_1[89] - total_stock_second_period_89 = 0
 R899: - 700 Sold_0[89] - 700 Sold_1[89] + 700 Lost_0[89] + 700 Lost_1[89]
   + 300 Left_1[89] + profits_vars[89] = 0
 Fraction_lower_bound_90: Fraction[90] >= 0
 Fraction_upper_bound_90: Fraction[90] <= 1
 R902: 0.6854758852755324 alphas[0] + 1.39751439667991 alphas[1]
   - 1.377787726371684 alphas[2] + 1.617838802351092 alphas[3] - alphas[4]
   + f_var[90] = 0
 R903: - 1857.770397056156 Fraction[90] + Q0_var[90] = 0
 R904: Q0_var[90] + Lost_0_expr_90 = 1780.119170907275
 R905: Q0_var[90] + Q_hat_adjusted_90 = 2216.624086318584
 R906: Sold_0[90] - Q0_var[90] - Q1_var[90] + total_stock_second_period_90
   = 0
 R907: total_stock_second_period_90 + Lost_1_expr_90 = 436.4388421171717
 R908: Sold_1[90] + Left_1[90] - total_stock_second_period_90 = 0
 R909: - 700 Sold_0[90] - 700 Sold_1[90] + 700 Lost_0[90] + 700 Lost_1[90]
   + 300 Left_1[90] + profits_vars[90] = 0
 Fraction_lower_bound_91: Fraction[91] >= 0
 Fraction_upper_bound_91: Fraction[91] <= 1
 R912: - 0.5618240306651073 alphas[0] + 0.1722843147501651 alphas[1]
   + 1.538515513468171 alphas[2] + 0.5670736714733345 alphas[3] - alphas[4]
   + f_var[91] = 0
 R913: - 1857.770397056156 Fraction[91] + Q0_var[91] = 0
 R914: Q0_var[91] + Lost_0_expr_91 = 377.6154961474439
 R915: Q0_var[91] + Q_hat_adjusted_91 = 439.985862736669
 R916: Sold_0[91] - Q0_var[91] - Q1_var[91] + total_stock_second_period_91
   = 0
 R917: total_stock_second_period_91 + Lost_1_expr_91 = 62.33601784942824
 R918: Sold_1[91] + Left_1[91] - total_stock_second_period_91 = 0
 R919: - 700 Sold_0[91] - 700 Sold_1[91] + 700 Lost_0[91] + 700 Lost_1[91]
   + 300 Left_1[91] + profits_vars[91] = 0
 Fraction_lower_bound_92: Fraction[92] >= 0
 Fraction_upper_bound_92: Fraction[92] <= 1
 R922: 1.353285229414931 alphas[0] + 0.7329783333920462 alphas[1]
   - 0.2214997541553104 alphas[2] + 0.8825216404453005 alphas[3]
   - alphas[4] + f_var[92] = 0
 R923: - 1857.770397056156 Fraction[92] + Q0_var[92] = 0
 R924: Q0_var[92] + Lost_0_expr_92 = 913.675091987402
 R925: Q0_var[92] + Q_hat_adjusted_92 = 1191.394506118208
 R926: Sold_0[92] - Q0_var[92] - Q1_var[92] + total_stock_second_period_92
   = 0
 R927: total_stock_second_period_92 + Lost_1_expr_92 = 277.6635287002167
 R928: Sold_1[92] + Left_1[92] - total_stock_second_period_92 = 0
 R929: - 700 Sold_0[92] - 700 Sold_1[92] + 700 Lost_0[92] + 700 Lost_1[92]
   + 300 Left_1[92] + profits_vars[92] = 0
 Fraction_lower_bound_93: Fraction[93] >= 0
 Fraction_upper_bound_93: Fraction[93] <= 1
 R932: - 0.7367285742325661 alphas[0] - 1.268906913246554 alphas[1]
   + 0.2332654945729538 alphas[2] - 0.0553975500098863 alphas[3]
   - alphas[4] + f_var[93] = 0
 R933: - 1857.770397056156 Fraction[93] + Q0_var[93] = 0
 R934: Q0_var[93] + Lost_0_expr_93 = 1178.211683657832
 R935: Q0_var[93] + Q_hat_adjusted_93 = 1473.740335279982
 R936: Sold_0[93] - Q0_var[93] - Q1_var[93] + total_stock_second_period_93
   = 0
 R937: total_stock_second_period_93 + Lost_1_expr_93 = 295.4587723077492
 R938: Sold_1[93] + Left_1[93] - total_stock_second_period_93 = 0
 R939: - 700 Sold_0[93] - 700 Sold_1[93] + 700 Lost_0[93] + 700 Lost_1[93]
   + 300 Left_1[93] + profits_vars[93] = 0
 Fraction_lower_bound_94: Fraction[94] >= 0
 Fraction_upper_bound_94: Fraction[94] <= 1
 R942: 0.7897491433787203 alphas[0] + 1.497237730159426 alphas[1]
   - 0.184147094643235 alphas[2] - 0.8333121373732969 alphas[3] - alphas[4]
   + f_var[94] = 0
 R943: - 1857.770397056156 Fraction[94] + Q0_var[94] = 0
 R944: Q0_var[94] + Lost_0_expr_94 = 1018.296817460222
 R945: Q0_var[94] + Q_hat_adjusted_94 = 1249.611737158467
 R946: Sold_0[94] - Q0_var[94] - Q1_var[94] + total_stock_second_period_94
   = 0
 R947: total_stock_second_period_94 + Lost_1_expr_94 = 231.2643991345874
 R948: Sold_1[94] + Left_1[94] - total_stock_second_period_94 = 0
 R949: - 700 Sold_0[94] - 700 Sold_1[94] + 700 Lost_0[94] + 700 Lost_1[94]
   + 300 Left_1[94] + profits_vars[94] = 0
 Fraction_lower_bound_95: Fraction[95] >= 0
 Fraction_upper_bound_95: Fraction[95] <= 1
 R952: 1.169525209306358 alphas[0] - 1.641727304325608 alphas[1]
   + 0.3694924180612342 alphas[2] + 1.328627568031479 alphas[3] - alphas[4]
   + f_var[95] = 0
 R953: - 1857.770397056156 Fraction[95] + Q0_var[95] = 0
 R954: Q0_var[95] + Lost_0_expr_95 = 883.1668838513926
 R955: Q0_var[95] + Q_hat_adjusted_95 = 1167.351297929215
 R956: Sold_0[95] - Q0_var[95] - Q1_var[95] + total_stock_second_period_95
   = 0
 R957: total_stock_second_period_95 + Lost_1_expr_95 = 284.1222614374472
 R958: Sold_1[95] + Left_1[95] - total_stock_second_period_95 = 0
 R959: - 700 Sold_0[95] - 700 Sold_1[95] + 700 Lost_0[95] + 700 Lost_1[95]
   + 300 Left_1[95] + profits_vars[95] = 0
 Fraction_lower_bound_96: Fraction[96] >= 0
 Fraction_upper_bound_96: Fraction[96] <= 1
 R962: - 0.2725711822879711 alphas[0] + 0.4051212194668198 alphas[1]
   + 0.9768970233144973 alphas[2] - 0.0826569852179595 alphas[3]
   - alphas[4] + f_var[96] = 0
 R963: - 1857.770397056156 Fraction[96] + Q0_var[96] = 0
 R964: Q0_var[96] + Lost_0_expr_96 = 556.6304201864218
 R965: Q0_var[96] + Q_hat_adjusted_96 = 680.542542576186
 R966: Sold_0[96] - Q0_var[96] - Q1_var[96] + total_stock_second_period_96
   = 0
 R967: total_stock_second_period_96 + Lost_1_expr_96 = 123.8651451030456
 R968: Sold_1[96] + Left_1[96] - total_stock_second_period_96 = 0
 R969: - 700 Sold_0[96] - 700 Sold_1[96] + 700 Lost_0[96] + 700 Lost_1[96]
   + 300 Left_1[96] + profits_vars[96] = 0
 Fraction_lower_bound_97: Fraction[97] >= 0
 Fraction_upper_bound_97: Fraction[97] <= 1
 R972: 1.7526395234911 alphas[0] + 0.4005839328483733 alphas[1]
   + 1.573551572131588 alphas[2] - 1.515061078153265 alphas[3] - alphas[4]
   + f_var[97] = 0
 R973: - 1857.770397056156 Fraction[97] + Q0_var[97] = 0
 R974: Q0_var[97] + Lost_0_expr_97 = 198.7501399662573
 R975: Q0_var[97] + Q_hat_adjusted_97 = 200.0931617535854
 R976: Sold_0[97] - Q0_var[97] - Q1_var[97] + total_stock_second_period_97
   = 0
 R977: total_stock_second_period_97 + Lost_1_expr_97 = 1.923033295570896
 R978: Sold_1[97] + Left_1[97] - total_stock_second_period_97 = 0
 R979: - 700 Sold_0[97] - 700 Sold_1[97] + 700 Lost_0[97] + 700 Lost_1[97]
   + 300 Left_1[97] + profits_vars[97] = 0
 Fraction_lower_bound_98: Fraction[98] >= 0
 Fraction_upper_bound_98: Fraction[98] <= 1
 R982: - 1.139381344808652 alphas[0] + 1.554803594034353 alphas[1]
   - 0.1631475788485993 alphas[2] + 0.8856364542056198 alphas[3]
   - alphas[4] + f_var[98] = 0
 R983: - 1857.770397056156 Fraction[98] + Q0_var[98] = 0
 R984: Q0_var[98] + Lost_0_expr_98 = 1272.512911022482
 R985: Q0_var[98] + Q_hat_adjusted_98 = 1540.689992308194
 R986: Sold_0[98] - Q0_var[98] - Q1_var[98] + total_stock_second_period_98
   = 0
 R987: total_stock_second_period_98 + Lost_1_expr_98 = 268.1198628484569
 R988: Sold_1[98] + Left_1[98] - total_stock_second_period_98 = 0
 R989: - 700 Sold_0[98] - 700 Sold_1[98] + 700 Lost_0[98] + 700 Lost_1[98]
   + 300 Left_1[98] + profits_vars[98] = 0
 Fraction_lower_bound_99: Fraction[99] >= 0
 Fraction_upper_bound_99: Fraction[99] <= 1
 R992: 1.807746148143607 alphas[0] + 0.9820524445090387 alphas[1]
   - 1.292905316829836 alphas[2] - 0.6313433192970277 alphas[3] - alphas[4]
   + f_var[99] = 0
 R993: - 1857.770397056156 Fraction[99] + Q0_var[99] = 0
 R994: Q0_var[99] + Lost_0_expr_99 = 1804.404463805455
 R995: Q0_var[99] + Q_hat_adjusted_99 = 2204.254164240796
 R996: Sold_0[99] - Q0_var[99] - Q1_var[99] + total_stock_second_period_99
   = 0
 R997: total_stock_second_period_99 + Lost_1_expr_99 = 399.7922801554564
 R998: Sold_1[99] + Left_1[99] - total_stock_second_period_99 = 0
 R999: - 700 Sold_0[99] - 700 Sold_1[99] + 700 Lost_0[99] + 700 Lost_1[99]
   + 300 Left_1[99] + profits_vars[99] = 0
 Fraction_lower_bound_100: Fraction[100] >= 0
 Fraction_upper_bound_100: Fraction[100] <= 1
 R1002: - 0.5990317632301936 alphas[0] + 0.2495064827590724 alphas[1]
   + 0.3412470739734619 alphas[2] - 0.3463409079010307 alphas[3]
   - alphas[4] + f_var[100] = 0
 R1003: - 1857.770397056156 Fraction[100] + Q0_var[100] = 0
 R1004: Q0_var[100] + Lost_0_expr_100 = 979.4206476561085
 R1005: Q0_var[100] + Q_hat_adjusted_100 = 1200.701028350692
 R1006: Sold_0[100] - Q0_var[100] - Q1_var[100]
   + total_stock_second_period_100 = 0
 R1007: total_stock_second_period_100 + Lost_1_expr_100
   = 221.2303651108812
 R1008: Sold_1[100] + Left_1[100] - total_stock_second_period_100 = 0
 R1009: - 700 Sold_0[100] - 700 Sold_1[100] + 700 Lost_0[100]
   + 700 Lost_1[100] + 300 Left_1[100] + profits_vars[100] = 0
 Fraction_lower_bound_101: Fraction[101] >= 0
 Fraction_upper_bound_101: Fraction[101] <= 1
 R1012: 0.8591080876772906 alphas[0] - 1.541044391641392 alphas[1]
   + 0.8644982682321272 alphas[2] + 1.624967040227427 alphas[3] - alphas[4]
   + f_var[101] = 0
 R1013: - 1857.770397056156 Fraction[101] + Q0_var[101] = 0
 R1014: Q0_var[101] + Lost_0_expr_101 = 992.0590712696317
 R1015: Q0_var[101] + Q_hat_adjusted_101 = 1208.520424570029
 R1016: Sold_0[101] - Q0_var[101] - Q1_var[101]
   + total_stock_second_period_101 = 0
 R1017: total_stock_second_period_101 + Lost_1_expr_101
   = 216.4184466869513
 R1018: Sold_1[101] + Left_1[101] - total_stock_second_period_101 = 0
 R1019: - 700 Sold_0[101] - 700 Sold_1[101] + 700 Lost_0[101]
   + 700 Lost_1[101] + 300 Left_1[101] + profits_vars[101] = 0
 Fraction_lower_bound_102: Fraction[102] >= 0
 Fraction_upper_bound_102: Fraction[102] <= 1
 R1022: - 0.8041878226317811 alphas[0] + 1.272380162811085 alphas[1]
   + 1.359837363939117 alphas[2] + 0.049580618882694 alphas[3] - alphas[4]
   + f_var[102] = 0
 R1023: - 1857.770397056156 Fraction[102] + Q0_var[102] = 0
 R1024: Q0_var[102] + Lost_0_expr_102 = 353.5273035153854
 R1025: Q0_var[102] + Q_hat_adjusted_102 = 399.2407998812629
 R1026: Sold_0[102] - Q0_var[102] - Q1_var[102]
   + total_stock_second_period_102 = 0
 R1027: total_stock_second_period_102 + Lost_1_expr_102
   = 45.66830291453748
 R1028: Sold_1[102] + Left_1[102] - total_stock_second_period_102 = 0
 R1029: - 700 Sold_0[102] - 700 Sold_1[102] + 700 Lost_0[102]
   + 700 Lost_1[102] + 300 Left_1[102] + profits_vars[102] = 0
 Fraction_lower_bound_103: Fraction[103] >= 0
 Fraction_upper_bound_103: Fraction[103] <= 1
 R1032: - 1.6158180566596 alphas[0] - 1.595334533246673 alphas[1]
   + 1.343205904950626 alphas[2] - 0.7383254957307984 alphas[3] - alphas[4]
   + f_var[103] = 0
 R1033: - 1857.770397056156 Fraction[103] + Q0_var[103] = 0
 R1034: Q0_var[103] + Lost_0_expr_103 = 640.2924529199969
 R1035: Q0_var[103] + Q_hat_adjusted_103 = 789.6485117800063
 R1036: Sold_0[103] - Q0_var[103] - Q1_var[103]
   + total_stock_second_period_103 = 0
 R1037: total_stock_second_period_103 + Lost_1_expr_103
   = 149.3021682856962
 R1038: Sold_1[103] + Left_1[103] - total_stock_second_period_103 = 0
 R1039: - 700 Sold_0[103] - 700 Sold_1[103] + 700 Lost_0[103]
   + 700 Lost_1[103] + 300 Left_1[103] + profits_vars[103] = 0
 Fraction_lower_bound_104: Fraction[104] >= 0
 Fraction_upper_bound_104: Fraction[104] <= 1
 R1042: 0.9351057908530026 alphas[0] - 1.338253409075937 alphas[1]
   + 0.8696049731826513 alphas[2] + 1.510264613614517 alphas[3] - alphas[4]
   + f_var[104] = 0
 R1043: - 1857.770397056156 Fraction[104] + Q0_var[104] = 0
 R1044: Q0_var[104] + Lost_0_expr_104 = 877.9304765979887
 R1045: Q0_var[104] + Q_hat_adjusted_104 = 1084.295503953547
 R1046: Sold_0[104] - Q0_var[104] - Q1_var[104]
   + total_stock_second_period_104 = 0
 R1047: total_stock_second_period_104 + Lost_1_expr_104
   = 206.3017517732627
 R1048: Sold_1[104] + Left_1[104] - total_stock_second_period_104 = 0
 R1049: - 700 Sold_0[104] - 700 Sold_1[104] + 700 Lost_0[104]
   + 700 Lost_1[104] + 300 Left_1[104] + profits_vars[104] = 0
 Fraction_lower_bound_105: Fraction[105] >= 0
 Fraction_upper_bound_105: Fraction[105] <= 1
 R1052: - 0.2355442167957314 alphas[0] + 0.0701132985264431 alphas[1]
   + 1.292036466784624 alphas[2] - 1.315870815031256 alphas[3] - alphas[4]
   + f_var[105] = 0
 R1053: - 1857.770397056156 Fraction[105] + Q0_var[105] = 0
 R1054: Q0_var[105] + Lost_0_expr_105 = 526.3605294604208
 R1055: Q0_var[105] + Q_hat_adjusted_105 = 601.8417766262833
 R1056: Sold_0[105] - Q0_var[105] - Q1_var[105]
   + total_stock_second_period_105 = 0
 R1057: total_stock_second_period_105 + Lost_1_expr_105
   = 75.44574864592298
 R1058: Sold_1[105] + Left_1[105] - total_stock_second_period_105 = 0
 R1059: - 700 Sold_0[105] - 700 Sold_1[105] + 700 Lost_0[105]
   + 700 Lost_1[105] + 300 Left_1[105] + profits_vars[105] = 0
 Fraction_lower_bound_106: Fraction[106] >= 0
 Fraction_upper_bound_106: Fraction[106] <= 1
 R1062: - 0.2923403847911335 alphas[0] + 0.5023340038546853 alphas[1]
   + 0.7177189799891818 alphas[2] - 0.1001074170967533 alphas[3]
   - alphas[4] + f_var[106] = 0
 R1063: - 1857.770397056156 Fraction[106] + Q0_var[106] = 0
 R1064: Q0_var[106] + Lost_0_expr_106 = 712.4491036870108
 R1065: Q0_var[106] + Q_hat_adjusted_106 = 870.1839963760254
 R1066: Sold_0[106] - Q0_var[106] - Q1_var[106]
   + total_stock_second_period_106 = 0
 R1067: total_stock_second_period_106 + Lost_1_expr_106 = 157.676172081327
 R1068: Sold_1[106] + Left_1[106] - total_stock_second_period_106 = 0
 R1069: - 700 Sold_0[106] - 700 Sold_1[106] + 700 Lost_0[106]
   + 700 Lost_1[106] + 300 Left_1[106] + profits_vars[106] = 0
 Fraction_lower_bound_107: Fraction[107] >= 0
 Fraction_upper_bound_107: Fraction[107] <= 1
 R1072: - 0.2215801620306154 alphas[0] + 0.8209403987877385 alphas[1]
   + 1.409891071548811 alphas[2] + 1.555298937498326 alphas[3] - alphas[4]
   + f_var[107] = 0
 R1073: - 1857.770397056156 Fraction[107] + Q0_var[107] = 0
 R1074: Q0_var[107] + Lost_0_expr_107 = 487.614092931941
 R1075: Q0_var[107] + Q_hat_adjusted_107 = 553.3975301827919
 R1076: Sold_0[107] - Q0_var[107] - Q1_var[107]
   + total_stock_second_period_107 = 0
 R1077: total_stock_second_period_107 + Lost_1_expr_107
   = 65.74145737201637
 R1078: Sold_1[107] + Left_1[107] - total_stock_second_period_107 = 0
 R1079: - 700 Sold_0[107] - 700 Sold_1[107] + 700 Lost_0[107]
   + 700 Lost_1[107] + 300 Left_1[107] + profits_vars[107] = 0
 Fraction_lower_bound_108: Fraction[108] >= 0
 Fraction_upper_bound_108: Fraction[108] <= 1
 R1082: 1.026895557883283 alphas[0] - 0.4734702076956678 alphas[1]
   + 1.411786355113067 alphas[2] + 0.8999538685915797 alphas[3] - alphas[4]
   + f_var[108] = 0
 R1083: - 1857.770397056156 Fraction[108] + Q0_var[108] = 0
 R1084: Q0_var[108] + Lost_0_expr_108 = 320.6134520234763
 R1085: Q0_var[108] + Q_hat_adjusted_108 = 410.2095445051348
 R1086: Sold_0[108] - Q0_var[108] - Q1_var[108]
   + total_stock_second_period_108 = 0
 R1087: total_stock_second_period_108 + Lost_1_expr_108
   = 89.55871571389466
 R1088: Sold_1[108] + Left_1[108] - total_stock_second_period_108 = 0
 R1089: - 700 Sold_0[108] - 700 Sold_1[108] + 700 Lost_0[108]
   + 700 Lost_1[108] + 300 Left_1[108] + profits_vars[108] = 0
 Fraction_lower_bound_109: Fraction[109] >= 0
 Fraction_upper_bound_109: Fraction[109] <= 1
 R1092: - 1.582066533652927 alphas[0] + 1.498017776753188 alphas[1]
   - 1.339885252174329 alphas[2] - 1.566688647791637 alphas[3] - alphas[4]
   + f_var[109] = 0
 R1093: - 1857.770397056156 Fraction[109] + Q0_var[109] = 0
 R1094: Q0_var[109] + Lost_0_expr_109 = 1946.685845362863
 R1095: Q0_var[109] + Q_hat_adjusted_109 = 2361.874727520631
 R1096: Sold_0[109] - Q0_var[109] - Q1_var[109]
   + total_stock_second_period_109 = 0
 R1097: total_stock_second_period_109 + Lost_1_expr_109
   = 415.1437698934867
 R1098: Sold_1[109] + Left_1[109] - total_stock_second_period_109 = 0
 R1099: - 700 Sold_0[109] - 700 Sold_1[109] + 700 Lost_0[109]
   + 700 Lost_1[109] + 300 Left_1[109] + profits_vars[109] = 0
 Fraction_lower_bound_110: Fraction[110] >= 0
 Fraction_upper_bound_110: Fraction[110] <= 1
 R1102: 0.2258159741969403 alphas[0] + 1.557253374769808 alphas[1]
   + 1.075896252882521 alphas[2] - 0.310621141424893 alphas[3] - alphas[4]
   + f_var[110] = 0
 R1103: - 1857.770397056156 Fraction[110] + Q0_var[110] = 0
 R1104: Q0_var[110] + Lost_0_expr_110 = 222.1762921283521
 R1105: Q0_var[110] + Q_hat_adjusted_110 = 282.8284361096541
 R1106: Sold_0[110] - Q0_var[110] - Q1_var[110]
   + total_stock_second_period_110 = 0
 R1107: total_stock_second_period_110 + Lost_1_expr_110
   = 60.59888041210703
 R1108: Sold_1[110] + Left_1[110] - total_stock_second_period_110 = 0
 R1109: - 700 Sold_0[110] - 700 Sold_1[110] + 700 Lost_0[110]
   + 700 Lost_1[110] + 300 Left_1[110] + profits_vars[110] = 0
 Fraction_lower_bound_111: Fraction[111] >= 0
 Fraction_upper_bound_111: Fraction[111] <= 1
 R1112: - 1.201841349680769 alphas[0] + 0.1561314342205306 alphas[1]
   - 0.3076881135538521 alphas[2] + 1.295493082193871 alphas[3] - alphas[4]
   + f_var[111] = 0
 R1113: - 1857.770397056156 Fraction[111] + Q0_var[111] = 0
 R1114: Q0_var[111] + Lost_0_expr_111 = 1121.570158585745
 R1115: Q0_var[111] + Q_hat_adjusted_111 = 1465.554325321557
 R1116: Sold_0[111] - Q0_var[111] - Q1_var[111]
   + total_stock_second_period_111 = 0
 R1117: total_stock_second_period_111 + Lost_1_expr_111
   = 343.9279303010536
 R1118: Sold_1[111] + Left_1[111] - total_stock_second_period_111 = 0
 R1119: - 700 Sold_0[111] - 700 Sold_1[111] + 700 Lost_0[111]
   + 700 Lost_1[111] + 300 Left_1[111] + profits_vars[111] = 0
 Fraction_lower_bound_112: Fraction[112] >= 0
 Fraction_upper_bound_112: Fraction[112] <= 1
 R1122: - 0.6764879648324882 alphas[0] + 1.37933488650665 alphas[1]
   - 1.05165525479443 alphas[2] + 0.6864211187519934 alphas[3] - alphas[4]
   + f_var[112] = 0
 R1123: - 1857.770397056156 Fraction[112] + Q0_var[112] = 0
 R1124: Q0_var[112] + Lost_0_expr_112 = 1729.40645281362
 R1125: Q0_var[112] + Q_hat_adjusted_112 = 2123.907348083298
 R1126: Sold_0[112] - Q0_var[112] - Q1_var[112]
   + total_stock_second_period_112 = 0
 R1127: total_stock_second_period_112 + Lost_1_expr_112
   = 394.4478485244075
 R1128: Sold_1[112] + Left_1[112] - total_stock_second_period_112 = 0
 R1129: - 700 Sold_0[112] - 700 Sold_1[112] + 700 Lost_0[112]
   + 700 Lost_1[112] + 300 Left_1[112] + profits_vars[112] = 0
 Fraction_lower_bound_113: Fraction[113] >= 0
 Fraction_upper_bound_113: Fraction[113] <= 1
 R1132: 0.7610344105417191 alphas[0] + 0.7558440446940994 alphas[1]
   + 0.0517190617205294 alphas[2] + 0.1142604441178019 alphas[3]
   - alphas[4] + f_var[113] = 0
 R1133: - 1857.770397056156 Fraction[113] + Q0_var[113] = 0
 R1134: Q0_var[113] + Lost_0_expr_113 = 853.1071013674194
 R1135: Q0_var[113] + Q_hat_adjusted_113 = 1088.303991590657
 R1136: Sold_0[113] - Q0_var[113] - Q1_var[113]
   + total_stock_second_period_113 = 0
 R1137: total_stock_second_period_113 + Lost_1_expr_113
   = 235.1408307602986
 R1138: Sold_1[113] + Left_1[113] - total_stock_second_period_113 = 0
 R1139: - 700 Sold_0[113] - 700 Sold_1[113] + 700 Lost_0[113]
   + 700 Lost_1[113] + 300 Left_1[113] + profits_vars[113] = 0
 Fraction_lower_bound_114: Fraction[114] >= 0
 Fraction_upper_bound_114: Fraction[114] <= 1
 R1142: - 1.085239619756792 alphas[0] + 0.8602189597761779 alphas[1]
   + 1.106352564289532 alphas[2] + 1.58814517252006 alphas[3] - alphas[4]
   + f_var[114] = 0
 R1143: - 1857.770397056156 Fraction[114] + Q0_var[114] = 0
 R1144: Q0_var[114] + Lost_0_expr_114 = 503.1413172655861
 R1145: Q0_var[114] + Q_hat_adjusted_114 = 621.2046790189715
 R1146: Sold_0[114] - Q0_var[114] - Q1_var[114]
   + total_stock_second_period_114 = 0
 R1147: total_stock_second_period_114 + Lost_1_expr_114 = 118.011455290041
 R1148: Sold_1[114] + Left_1[114] - total_stock_second_period_114 = 0
 R1149: - 700 Sold_0[114] - 700 Sold_1[114] + 700 Lost_0[114]
   + 700 Lost_1[114] + 300 Left_1[114] + profits_vars[114] = 0
 Fraction_lower_bound_115: Fraction[115] >= 0
 Fraction_upper_bound_115: Fraction[115] <= 1
 R1152: 0.4068090127431765 alphas[0] + 0.7516046105999766 alphas[1]
   + 0.9454737660430766 alphas[2] + 0.2667656367146409 alphas[3]
   - alphas[4] + f_var[115] = 0
 R1153: - 1857.770397056156 Fraction[115] + Q0_var[115] = 0
 R1154: Q0_var[115] + Lost_0_expr_115 = 614.6329623680242
 R1155: Q0_var[115] + Q_hat_adjusted_115 = 727.3380437434951
 R1156: Sold_0[115] - Q0_var[115] - Q1_var[115]
   + total_stock_second_period_115 = 0
 R1157: total_stock_second_period_115 + Lost_1_expr_115
   = 112.6504198975765
 R1158: Sold_1[115] + Left_1[115] - total_stock_second_period_115 = 0
 R1159: - 700 Sold_0[115] - 700 Sold_1[115] + 700 Lost_0[115]
   + 700 Lost_1[115] + 300 Left_1[115] + profits_vars[115] = 0
 Fraction_lower_bound_116: Fraction[116] >= 0
 Fraction_upper_bound_116: Fraction[116] <= 1
 R1162: - 1.325893352264612 alphas[0] + 1.165766420957298 alphas[1]
   + 0.1332433239495343 alphas[2] + 0.00347050870662119 alphas[3]
   - alphas[4] + f_var[116] = 0
 R1163: - 1857.770397056156 Fraction[116] + Q0_var[116] = 0
 R1164: Q0_var[116] + Lost_0_expr_116 = 799.040100767827
 R1165: Q0_var[116] + Q_hat_adjusted_116 = 1030.049386806224
 R1166: Sold_0[116] - Q0_var[116] - Q1_var[116]
   + total_stock_second_period_116 = 0
 R1167: total_stock_second_period_116 + Lost_1_expr_116 = 230.958309854928
 R1168: Sold_1[116] + Left_1[116] - total_stock_second_period_116 = 0
 R1169: - 700 Sold_0[116] - 700 Sold_1[116] + 700 Lost_0[116]
   + 700 Lost_1[116] + 300 Left_1[116] + profits_vars[116] = 0
 Fraction_lower_bound_117: Fraction[117] >= 0
 Fraction_upper_bound_117: Fraction[117] <= 1
 R1172: - 0.2538350787350102 alphas[0] + 1.569358671263482 alphas[1]
   - 0.1982425846086925 alphas[2] + 0.8347947823575763 alphas[3]
   - alphas[4] + f_var[117] = 0
 R1173: - 1857.770397056156 Fraction[117] + Q0_var[117] = 0
 R1174: Q0_var[117] + Lost_0_expr_117 = 985.5575272988551
 R1175: Q0_var[117] + Q_hat_adjusted_117 = 1247.759038099171
 R1176: Sold_0[117] - Q0_var[117] - Q1_var[117]
   + total_stock_second_period_117 = 0
 R1177: total_stock_second_period_117 + Lost_1_expr_117
   = 262.1389877326196
 R1178: Sold_1[117] + Left_1[117] - total_stock_second_period_117 = 0
 R1179: - 700 Sold_0[117] - 700 Sold_1[117] + 700 Lost_0[117]
   + 700 Lost_1[117] + 300 Left_1[117] + profits_vars[117] = 0
 Fraction_lower_bound_118: Fraction[118] >= 0
 Fraction_upper_bound_118: Fraction[118] <= 1
 R1182: - 1.328153688822942 alphas[0] + 1.218567300515444 alphas[1]
   + 0.4239904563503471 alphas[2] - 0.3319953134510593 alphas[3]
   - alphas[4] + f_var[118] = 0
 R1183: - 1857.770397056156 Fraction[118] + Q0_var[118] = 0
 R1184: Q0_var[118] + Lost_0_expr_118 = 965.5760113542973
 R1185: Q0_var[118] + Q_hat_adjusted_118 = 1149.009253985167
 R1186: Sold_0[118] - Q0_var[118] - Q1_var[118]
   + total_stock_second_period_118 = 0
 R1187: total_stock_second_period_118 + Lost_1_expr_118 = 183.37591264488
 R1188: Sold_1[118] + Left_1[118] - total_stock_second_period_118 = 0
 R1189: - 700 Sold_0[118] - 700 Sold_1[118] + 700 Lost_0[118]
   + 700 Lost_1[118] + 300 Left_1[118] + profits_vars[118] = 0
 Fraction_lower_bound_119: Fraction[119] >= 0
 Fraction_upper_bound_119: Fraction[119] <= 1
 R1192: - 0.6516461698733702 alphas[0] - 0.487086907321053 alphas[1]
   - 1.079580343429847 alphas[2] - 0.8887390791507166 alphas[3] - alphas[4]
   + f_var[119] = 0
 R1193: - 1857.770397056156 Fraction[119] + Q0_var[119] = 0
 R1194: Q0_var[119] + Lost_0_expr_119 = 1758.06134228584
 R1195: Q0_var[119] + Q_hat_adjusted_119 = 2203.720114625233
 R1196: Sold_0[119] - Q0_var[119] - Q1_var[119]
   + total_stock_second_period_119 = 0
 R1197: total_stock_second_period_119 + Lost_1_expr_119
   = 445.6028865797322
 R1198: Sold_1[119] + Left_1[119] - total_stock_second_period_119 = 0
 R1199: - 700 Sold_0[119] - 700 Sold_1[119] + 700 Lost_0[119]
   + 700 Lost_1[119] + 300 Left_1[119] + profits_vars[119] = 0
 Fraction_lower_bound_120: Fraction[120] >= 0
 Fraction_upper_bound_120: Fraction[120] <= 1
 R1202: - 0.768647778470864 alphas[0] - 1.693519557273468 alphas[1]
   - 0.9735814552598103 alphas[2] + 0.861325002266787 alphas[3] - alphas[4]
   + f_var[120] = 0
 R1203: - 1857.770397056156 Fraction[120] + Q0_var[120] = 0
 R1204: Q0_var[120] + Lost_0_expr_120 = 1891.238243961441
 R1205: Q0_var[120] + Q_hat_adjusted_120 = 2386.394746390421
 R1206: Sold_0[120] - Q0_var[120] - Q1_var[120]
   + total_stock_second_period_120 = 0
 R1207: total_stock_second_period_120 + Lost_1_expr_120
   = 495.0721686963736
 R1208: Sold_1[120] + Left_1[120] - total_stock_second_period_120 = 0
 R1209: - 700 Sold_0[120] - 700 Sold_1[120] + 700 Lost_0[120]
   + 700 Lost_1[120] + 300 Left_1[120] + profits_vars[120] = 0
 Fraction_lower_bound_121: Fraction[121] >= 0
 Fraction_upper_bound_121: Fraction[121] <= 1
 R1212: 0.0320247357148636 alphas[0] - 1.748076476753906 alphas[1]
   - 1.586753387847609 alphas[2] - 0.4397610292208876 alphas[3] - alphas[4]
   + f_var[121] = 0
 R1213: - 1857.770397056156 Fraction[121] + Q0_var[121] = 0
 R1214: Q0_var[121] + Lost_0_expr_121 = 2295.92918895746
 R1215: Q0_var[121] + Q_hat_adjusted_121 = 2856.755635656255
 R1216: Sold_0[121] - Q0_var[121] - Q1_var[121]
   + total_stock_second_period_121 = 0
 R1217: total_stock_second_period_121 + Lost_1_expr_121
   = 560.7540796411829
 R1218: Sold_1[121] + Left_1[121] - total_stock_second_period_121 = 0
 R1219: - 700 Sold_0[121] - 700 Sold_1[121] + 700 Lost_0[121]
   + 700 Lost_1[121] + 300 Left_1[121] + profits_vars[121] = 0
 Fraction_lower_bound_122: Fraction[122] >= 0
 Fraction_upper_bound_122: Fraction[122] <= 1
 R1222: - 1.593989645178115 alphas[0] + 0.2230750916142635 alphas[1]
   + 1.539342700843575 alphas[2] - 0.5044400855967826 alphas[3] - alphas[4]
   + f_var[122] = 0
 R1223: - 1857.770397056156 Fraction[122] + Q0_var[122] = 0
 R1224: Q0_var[122] + Lost_0_expr_122 = 125.6735446919541
 R1225: Q0_var[122] + Q_hat_adjusted_122 = 185.038590428369
 R1226: Sold_0[122] - Q0_var[122] - Q1_var[122]
   + total_stock_second_period_122 = 0
 R1227: total_stock_second_period_122 + Lost_1_expr_122
   = 59.29501813212444
 R1228: Sold_1[122] + Left_1[122] - total_stock_second_period_122 = 0
 R1229: - 700 Sold_0[122] - 700 Sold_1[122] + 700 Lost_0[122]
   + 700 Lost_1[122] + 300 Left_1[122] + profits_vars[122] = 0
 Fraction_lower_bound_123: Fraction[123] >= 0
 Fraction_upper_bound_123: Fraction[123] <= 1
 R1232: - 0.478084006811853 alphas[0] + 1.05759657475438 alphas[1]
   - 1.477056592212199 alphas[2] - 1.549318782210023 alphas[3] - alphas[4]
   + f_var[123] = 0
 R1233: - 1857.770397056156 Fraction[123] + Q0_var[123] = 0
 R1234: Q0_var[123] + Lost_0_expr_123 = 2042.760817796573
 R1235: Q0_var[123] + Q_hat_adjusted_123 = 2481.115656905153
 R1236: Sold_0[123] - Q0_var[123] - Q1_var[123]
   + total_stock_second_period_123 = 0
 R1237: total_stock_second_period_123 + Lost_1_expr_123
   = 438.2919691561743
 R1238: Sold_1[123] + Left_1[123] - total_stock_second_period_123 = 0
 R1239: - 700 Sold_0[123] - 700 Sold_1[123] + 700 Lost_0[123]
   + 700 Lost_1[123] + 300 Left_1[123] + profits_vars[123] = 0
 Fraction_lower_bound_124: Fraction[124] >= 0
 Fraction_upper_bound_124: Fraction[124] <= 1
 R1242: 0.3090201994922049 alphas[0] - 0.5558612608809879 alphas[1]
   + 0.2738959010234472 alphas[2] - 0.9734205414419254 alphas[3]
   - alphas[4] + f_var[124] = 0
 R1243: - 1857.770397056156 Fraction[124] + Q0_var[124] = 0
 R1244: Q0_var[124] + Lost_0_expr_124 = 1086.899102552989
 R1245: Q0_var[124] + Q_hat_adjusted_124 = 1328.795658287675
 R1246: Sold_0[124] - Q0_var[124] - Q1_var[124]
   + total_stock_second_period_124 = 0
 R1247: total_stock_second_period_124 + Lost_1_expr_124
   = 241.8327568669766
 R1248: Sold_1[124] + Left_1[124] - total_stock_second_period_124 = 0
 R1249: - 700 Sold_0[124] - 700 Sold_1[124] + 700 Lost_0[124]
   + 700 Lost_1[124] + 300 Left_1[124] + profits_vars[124] = 0
 Fraction_lower_bound_125: Fraction[125] >= 0
 Fraction_upper_bound_125: Fraction[125] <= 1
 R1252: - 0.3436541121419401 alphas[0] - 0.0524470772033182 alphas[1]
   - 1.416784216055075 alphas[2] - 1.210420971009575 alphas[3] - alphas[4]
   + f_var[125] = 0
 R1253: - 1857.770397056156 Fraction[125] + Q0_var[125] = 0
 R1254: Q0_var[125] + Lost_0_expr_125 = 1757.520006066862
 R1255: Q0_var[125] + Q_hat_adjusted_125 = 2229.008261539172
 R1256: Sold_0[125] - Q0_var[125] - Q1_var[125]
   + total_stock_second_period_125 = 0
 R1257: total_stock_second_period_125 + Lost_1_expr_125
   = 471.4186035508104
 R1258: Sold_1[125] + Left_1[125] - total_stock_second_period_125 = 0
 R1259: - 700 Sold_0[125] - 700 Sold_1[125] + 700 Lost_0[125]
   + 700 Lost_1[125] + 300 Left_1[125] + profits_vars[125] = 0
 Fraction_lower_bound_126: Fraction[126] >= 0
 Fraction_upper_bound_126: Fraction[126] <= 1
 R1262: 1.755908819915272 alphas[0] - 1.744905140715958 alphas[1]
   - 0.7639468198331167 alphas[2] + 2.1911677782501731e-04 alphas[3]
   - alphas[4] + f_var[126] = 0
 R1263: - 1857.770397056156 Fraction[126] + Q0_var[126] = 0
 R1264: Q0_var[126] + Lost_0_expr_126 = 1983.283373163777
 R1265: Q0_var[126] + Q_hat_adjusted_126 = 2411.692796556808
 R1266: Sold_0[126] - Q0_var[126] - Q1_var[126]
   + total_stock_second_period_126 = 0
 R1267: total_stock_second_period_126 + Lost_1_expr_126 = 428.346289448767
 R1268: Sold_1[126] + Left_1[126] - total_stock_second_period_126 = 0
 R1269: - 700 Sold_0[126] - 700 Sold_1[126] + 700 Lost_0[126]
   + 700 Lost_1[126] + 300 Left_1[126] + profits_vars[126] = 0
 Fraction_lower_bound_127: Fraction[127] >= 0
 Fraction_upper_bound_127: Fraction[127] <= 1
 R1272: 0.7462392650065811 alphas[0] + 1.388727476815718 alphas[1]
   - 1.79604027412115 alphas[2] + 1.032086682666757 alphas[3] - alphas[4]
   + f_var[127] = 0
 R1273: - 1857.770397056156 Fraction[127] + Q0_var[127] = 0
 R1274: Q0_var[127] + Lost_0_expr_127 = 1573.932562604395
 R1275: Q0_var[127] + Q_hat_adjusted_127 = 2063.231533998986
 R1276: Sold_0[127] - Q0_var[127] - Q1_var[127]
   + total_stock_second_period_127 = 0
 R1277: total_stock_second_period_127 + Lost_1_expr_127
   = 489.2337290844525
 R1278: Sold_1[127] + Left_1[127] - total_stock_second_period_127 = 0
 R1279: - 700 Sold_0[127] - 700 Sold_1[127] + 700 Lost_0[127]
   + 700 Lost_1[127] + 300 Left_1[127] + profits_vars[127] = 0
 Fraction_lower_bound_128: Fraction[128] >= 0
 Fraction_upper_bound_128: Fraction[128] <= 1
 R1282: - 0.5359483396093765 alphas[0] - 1.045766656964326 alphas[1]
   - 1.002330464269174 alphas[2] - 1.709220835654117 alphas[3] - alphas[4]
   + f_var[128] = 0
 R1283: - 1857.770397056156 Fraction[128] + Q0_var[128] = 0
 R1284: Q0_var[128] + Lost_0_expr_128 = 1719.141299943697
 R1285: Q0_var[128] + Q_hat_adjusted_128 = 2162.493641503759
 R1286: Sold_0[128] - Q0_var[128] - Q1_var[128]
   + total_stock_second_period_128 = 0
 R1287: total_stock_second_period_128 + Lost_1_expr_128
   = 443.2774847387618
 R1288: Sold_1[128] + Left_1[128] - total_stock_second_period_128 = 0
 R1289: - 700 Sold_0[128] - 700 Sold_1[128] + 700 Lost_0[128]
   + 700 Lost_1[128] + 300 Left_1[128] + profits_vars[128] = 0
 Fraction_lower_bound_129: Fraction[129] >= 0
 Fraction_upper_bound_129: Fraction[129] <= 1
 R1292: 0.7873481106514182 alphas[0] + 0.6322170010787568 alphas[1]
   + 0.3714247868744201 alphas[2] + 1.22147577612443 alphas[3] - alphas[4]
   + f_var[129] = 0
 R1293: - 1857.770397056156 Fraction[129] + Q0_var[129] = 0
 R1294: Q0_var[129] + Lost_0_expr_129 = 835.2110498753746
 R1295: Q0_var[129] + Q_hat_adjusted_129 = 1041.651491681125
 R1296: Sold_0[129] - Q0_var[129] - Q1_var[129]
   + total_stock_second_period_129 = 0
 R1297: total_stock_second_period_129 + Lost_1_expr_129
   = 206.3828030264905
 R1298: Sold_1[129] + Left_1[129] - total_stock_second_period_129 = 0
 R1299: - 700 Sold_0[129] - 700 Sold_1[129] + 700 Lost_0[129]
   + 700 Lost_1[129] + 300 Left_1[129] + profits_vars[129] = 0
 Fraction_lower_bound_130: Fraction[130] >= 0
 Fraction_upper_bound_130: Fraction[130] <= 1
 R1302: - 0.3852099208511157 alphas[0] + 0.7915252884560517 alphas[1]
   - 0.1036831414198252 alphas[2] + 0.0643578255412784 alphas[3]
   - alphas[4] + f_var[130] = 0
 R1303: - 1857.770397056156 Fraction[130] + Q0_var[130] = 0
 R1304: Q0_var[130] + Lost_0_expr_130 = 1093.975868468196
 R1305: Q0_var[130] + Q_hat_adjusted_130 = 1362.471756091819
 R1306: Sold_0[130] - Q0_var[130] - Q1_var[130]
   + total_stock_second_period_130 = 0
 R1307: total_stock_second_period_130 + Lost_1_expr_130
   = 268.4379417137961
 R1308: Sold_1[130] + Left_1[130] - total_stock_second_period_130 = 0
 R1309: - 700 Sold_0[130] - 700 Sold_1[130] + 700 Lost_0[130]
   + 700 Lost_1[130] + 300 Left_1[130] + profits_vars[130] = 0
 Fraction_lower_bound_131: Fraction[131] >= 0
 Fraction_upper_bound_131: Fraction[131] <= 1
 R1312: 0.2914511904048849 alphas[0] - 0.6363729029570661 alphas[1]
   + 0.3302037963275837 alphas[2] + 1.428689464405712 alphas[3] - alphas[4]
   + f_var[131] = 0
 R1313: - 1857.770397056156 Fraction[131] + Q0_var[131] = 0
 R1314: Q0_var[131] + Lost_0_expr_131 = 1209.684516978956
 R1315: Q0_var[131] + Q_hat_adjusted_131 = 1474.942738349977
 R1316: Sold_0[131] - Q0_var[131] - Q1_var[131]
   + total_stock_second_period_131 = 0
 R1317: total_stock_second_period_131 + Lost_1_expr_131
   = 265.1959467506913
 R1318: Sold_1[131] + Left_1[131] - total_stock_second_period_131 = 0
 R1319: - 700 Sold_0[131] - 700 Sold_1[131] + 700 Lost_0[131]
   + 700 Lost_1[131] + 300 Left_1[131] + profits_vars[131] = 0
 Fraction_lower_bound_132: Fraction[132] >= 0
 Fraction_upper_bound_132: Fraction[132] <= 1
 R1322: 1.340140786533409 alphas[0] + 0.77577561592904 alphas[1]
   + 0.3701468330871368 alphas[2] - 1.533457294760165 alphas[3] - alphas[4]
   + f_var[132] = 0
 R1323: - 1857.770397056156 Fraction[132] + Q0_var[132] = 0
 R1324: Q0_var[132] + Lost_0_expr_132 = 830.0498175856582
 R1325: Q0_var[132] + Q_hat_adjusted_132 = 992.8249553482939
 R1326: Sold_0[132] - Q0_var[132] - Q1_var[132]
   + total_stock_second_period_132 = 0
 R1327: total_stock_second_period_132 + Lost_1_expr_132
   = 162.7222249858481
 R1328: Sold_1[132] + Left_1[132] - total_stock_second_period_132 = 0
 R1329: - 700 Sold_0[132] - 700 Sold_1[132] + 700 Lost_0[132]
   + 700 Lost_1[132] + 300 Left_1[132] + profits_vars[132] = 0
 Fraction_lower_bound_133: Fraction[133] >= 0
 Fraction_upper_bound_133: Fraction[133] <= 1
 R1332: 0.7580117297213627 alphas[0] - 0.6477481345165721 alphas[1]
   + 0.7321101808011132 alphas[2] - 1.604737882978622 alphas[3] - alphas[4]
   + f_var[133] = 0
 R1333: - 1857.770397056156 Fraction[133] + Q0_var[133] = 0
 R1334: Q0_var[133] + Lost_0_expr_133 = 776.809655078848
 R1335: Q0_var[133] + Q_hat_adjusted_133 = 943.6333635968746
 R1336: Sold_0[133] - Q0_var[133] - Q1_var[133]
   + total_stock_second_period_133 = 0
 R1337: total_stock_second_period_133 + Lost_1_expr_133
   = 166.7634357033703
 R1338: Sold_1[133] + Left_1[133] - total_stock_second_period_133 = 0
 R1339: - 700 Sold_0[133] - 700 Sold_1[133] + 700 Lost_0[133]
   + 700 Lost_1[133] + 300 Left_1[133] + profits_vars[133] = 0
 Fraction_lower_bound_134: Fraction[134] >= 0
 Fraction_upper_bound_134: Fraction[134] <= 1
 R1342: - 0.2134028970670837 alphas[0] - 0.14401455506691 alphas[1]
   - 0.0859590554569247 alphas[2] - 0.7742418913144581 alphas[3]
   - alphas[4] + f_var[134] = 0
 R1343: - 1857.770397056156 Fraction[134] + Q0_var[134] = 0
 R1344: Q0_var[134] + Lost_0_expr_134 = 1095.598053054835
 R1345: Q0_var[134] + Q_hat_adjusted_134 = 1382.820108599895
 R1346: Sold_0[134] - Q0_var[134] - Q1_var[134]
   + total_stock_second_period_134 = 0
 R1347: total_stock_second_period_134 + Lost_1_expr_134
   = 287.1590002003493
 R1348: Sold_1[134] + Left_1[134] - total_stock_second_period_134 = 0
 R1349: - 700 Sold_0[134] - 700 Sold_1[134] + 700 Lost_0[134]
   + 700 Lost_1[134] + 300 Left_1[134] + profits_vars[134] = 0
 Fraction_lower_bound_135: Fraction[135] >= 0
 Fraction_upper_bound_135: Fraction[135] <= 1
 R1352: - 0.2881599590812688 alphas[0] + 0.1720919216979848 alphas[1]
   - 0.7324692582919994 alphas[2] + 0.4751970099944423 alphas[3]
   - alphas[4] + f_var[135] = 0
 R1353: - 1857.770397056156 Fraction[135] + Q0_var[135] = 0
 R1354: Q0_var[135] + Lost_0_expr_135 = 1519.458779768865
 R1355: Q0_var[135] + Q_hat_adjusted_135 = 1903.972348149696
 R1356: Sold_0[135] - Q0_var[135] - Q1_var[135]
   + total_stock_second_period_135 = 0
 R1357: total_stock_second_period_135 + Lost_1_expr_135
   = 384.4696443531508
 R1358: Sold_1[135] + Left_1[135] - total_stock_second_period_135 = 0
 R1359: - 700 Sold_0[135] - 700 Sold_1[135] + 700 Lost_0[135]
   + 700 Lost_1[135] + 300 Left_1[135] + profits_vars[135] = 0
 Fraction_lower_bound_136: Fraction[136] >= 0
 Fraction_upper_bound_136: Fraction[136] <= 1
 R1362: - 0.2289935035057982 alphas[0] - 0.270767353364338 alphas[1]
   + 0.6749108790968259 alphas[2] + 0.7985879931843309 alphas[3]
   - alphas[4] + f_var[136] = 0
 R1363: - 1857.770397056156 Fraction[136] + Q0_var[136] = 0
 R1364: Q0_var[136] + Lost_0_expr_136 = 1011.725188404891
 R1365: Q0_var[136] + Q_hat_adjusted_136 = 1212.874935264606
 R1366: Sold_0[136] - Q0_var[136] - Q1_var[136]
   + total_stock_second_period_136 = 0
 R1367: total_stock_second_period_136 + Lost_1_expr_136
   = 201.1041831027184
 R1368: Sold_1[136] + Left_1[136] - total_stock_second_period_136 = 0
 R1369: - 700 Sold_0[136] - 700 Sold_1[136] + 700 Lost_0[136]
   + 700 Lost_1[136] + 300 Left_1[136] + profits_vars[136] = 0
 Fraction_lower_bound_137: Fraction[137] >= 0
 Fraction_upper_bound_137: Fraction[137] <= 1
 R1372: - 0.5110170400710127 alphas[0] + 0.636784587823516 alphas[1]
   - 0.1848290365875619 alphas[2] + 0.7616492648494492 alphas[3]
   - alphas[4] + f_var[137] = 0
 R1373: - 1857.770397056156 Fraction[137] + Q0_var[137] = 0
 R1374: Q0_var[137] + Lost_0_expr_137 = 1238.20144609854
 R1375: Q0_var[137] + Q_hat_adjusted_137 = 1533.48924164003
 R1376: Sold_0[137] - Q0_var[137] - Q1_var[137]
   + total_stock_second_period_137 = 0
 R1377: total_stock_second_period_137 + Lost_1_expr_137
   = 295.2257842432172
 R1378: Sold_1[137] + Left_1[137] - total_stock_second_period_137 = 0
 R1379: - 700 Sold_0[137] - 700 Sold_1[137] + 700 Lost_0[137]
   + 700 Lost_1[137] + 300 Left_1[137] + profits_vars[137] = 0
 Fraction_lower_bound_138: Fraction[138] >= 0
 Fraction_upper_bound_138: Fraction[138] <= 1
 R1382: - 0.5070926958433236 alphas[0] - 0.7858142829732103 alphas[1]
   + 1.231599000284186 alphas[2] + 1.228449352349613 alphas[3] - alphas[4]
   + f_var[138] = 0
 R1383: - 1857.770397056156 Fraction[138] + Q0_var[138] = 0
 R1384: Q0_var[138] + Lost_0_expr_138 = 565.4164114734263
 R1385: Q0_var[138] + Q_hat_adjusted_138 = 713.0448831537217
 R1386: Sold_0[138] - Q0_var[138] - Q1_var[138]
   + total_stock_second_period_138 = 0
 R1387: total_stock_second_period_138 + Lost_1_expr_138
   = 147.5452781226486
 R1388: Sold_1[138] + Left_1[138] - total_stock_second_period_138 = 0
 R1389: - 700 Sold_0[138] - 700 Sold_1[138] + 700 Lost_0[138]
   + 700 Lost_1[138] + 300 Left_1[138] + profits_vars[138] = 0
 Fraction_lower_bound_139: Fraction[139] >= 0
 Fraction_upper_bound_139: Fraction[139] <= 1
 R1392: 0.2819769326368924 alphas[0] + 0.2033979439565733 alphas[1]
   + 1.083813551594214 alphas[2] - 0.1194801298700888 alphas[3] - alphas[4]
   + f_var[139] = 0
 R1393: - 1857.770397056156 Fraction[139] + Q0_var[139] = 0
 R1394: Q0_var[139] + Lost_0_expr_139 = 540.9048249119948
 R1395: Q0_var[139] + Q_hat_adjusted_139 = 649.9607748805406
 R1396: Sold_0[139] - Q0_var[139] - Q1_var[139]
   + total_stock_second_period_139 = 0
 R1397: total_stock_second_period_139 + Lost_1_expr_139 = 109.003643542874
 R1398: Sold_1[139] + Left_1[139] - total_stock_second_period_139 = 0
 R1399: - 700 Sold_0[139] - 700 Sold_1[139] + 700 Lost_0[139]
   + 700 Lost_1[139] + 300 Left_1[139] + profits_vars[139] = 0
 Fraction_lower_bound_140: Fraction[140] >= 0
 Fraction_upper_bound_140: Fraction[140] <= 1
 R1402: - 1.381111994406633 alphas[0] + 0.387567041521336 alphas[1]
   + 1.477208398745087 alphas[2] + 1.179326180610298 alphas[3] - alphas[4]
   + f_var[140] = 0
 R1403: - 1857.770397056156 Fraction[140] + Q0_var[140] = 0
 R1404: Q0_var[140] + Lost_0_expr_140 = 720.2539137797929
 R1405: Q0_var[140] + Q_hat_adjusted_140 = 800.2125317572663
 R1406: Sold_0[140] - Q0_var[140] - Q1_var[140]
   + total_stock_second_period_140 = 0
 R1407: total_stock_second_period_140 + Lost_1_expr_140
   = 79.93161392167877
 R1408: Sold_1[140] + Left_1[140] - total_stock_second_period_140 = 0
 R1409: - 700 Sold_0[140] - 700 Sold_1[140] + 700 Lost_0[140]
   + 700 Lost_1[140] + 300 Left_1[140] + profits_vars[140] = 0
 Fraction_lower_bound_141: Fraction[141] >= 0
 Fraction_upper_bound_141: Fraction[141] <= 1
 R1412: 0.5102992658135936 alphas[0] - 1.199793813299203 alphas[1]
   - 1.738209173839643 alphas[2] + 0.5877023793369484 alphas[3] - alphas[4]
   + f_var[141] = 0
 R1413: - 1857.770397056156 Fraction[141] + Q0_var[141] = 0
 R1414: Q0_var[141] + Lost_0_expr_141 = 2459.008121895637
 R1415: Q0_var[141] + Q_hat_adjusted_141 = 3028.995656053919
 R1416: Sold_0[141] - Q0_var[141] - Q1_var[141]
   + total_stock_second_period_141 = 0
 R1417: total_stock_second_period_141 + Lost_1_expr_141
   = 569.9115957732306
 R1418: Sold_1[141] + Left_1[141] - total_stock_second_period_141 = 0
 R1419: - 700 Sold_0[141] - 700 Sold_1[141] + 700 Lost_0[141]
   + 700 Lost_1[141] + 300 Left_1[141] + profits_vars[141] = 0
 Fraction_lower_bound_142: Fraction[142] >= 0
 Fraction_upper_bound_142: Fraction[142] <= 1
 R1422: 0.2660782865592136 alphas[0] - 1.526380037853322 alphas[1]
   + 1.626667846434663 alphas[2] - 0.460754035406889 alphas[3] - alphas[4]
   + f_var[142] = 0
 R1423: - 1857.770397056156 Fraction[142] + Q0_var[142] = 0
 R1424: Q0_var[142] + Lost_0_expr_142 = 356.6318113686676
 R1425: Q0_var[142] + Q_hat_adjusted_142 = 445.0396435237016
 R1426: Sold_0[142] - Q0_var[142] - Q1_var[142]
   + total_stock_second_period_142 = 0
 R1427: total_stock_second_period_142 + Lost_1_expr_142 = 88.3663385028313
 R1428: Sold_1[142] + Left_1[142] - total_stock_second_period_142 = 0
 R1429: - 700 Sold_0[142] - 700 Sold_1[142] + 700 Lost_0[142]
   + 700 Lost_1[142] + 300 Left_1[142] + profits_vars[142] = 0
 Fraction_lower_bound_143: Fraction[143] >= 0
 Fraction_upper_bound_143: Fraction[143] <= 1
 R1432: - 1.364581366302974 alphas[0] + 1.454163056370584 alphas[1]
   + 1.018613651800408 alphas[2] - 0.8018236814647371 alphas[3] - alphas[4]
   + f_var[143] = 0
 R1433: - 1857.770397056156 Fraction[143] + Q0_var[143] = 0
 R1434: Q0_var[143] + Lost_0_expr_143 = 826.053118695464
 R1435: Q0_var[143] + Q_hat_adjusted_143 = 910.5643817360045
 R1436: Sold_0[143] - Q0_var[143] - Q1_var[143]
   + total_stock_second_period_143 = 0
 R1437: total_stock_second_period_143 + Lost_1_expr_143 = 84.4598476537488
 R1438: Sold_1[143] + Left_1[143] - total_stock_second_period_143 = 0
 R1439: - 700 Sold_0[143] - 700 Sold_1[143] + 700 Lost_0[143]
   + 700 Lost_1[143] + 300 Left_1[143] + profits_vars[143] = 0
 Fraction_lower_bound_144: Fraction[144] >= 0
 Fraction_upper_bound_144: Fraction[144] <= 1
 R1442: - 1.058051744147585 alphas[0] + 0.8213376072994766 alphas[1]
   - 0.4927787524859275 alphas[2] + 1.576906752471146 alphas[3] - alphas[4]
   + f_var[144] = 0
 R1443: - 1857.770397056156 Fraction[144] + Q0_var[144] = 0
 R1444: Q0_var[144] + Lost_0_expr_144 = 1359.37020043605
 R1445: Q0_var[144] + Q_hat_adjusted_144 = 1708.13811531989
 R1446: Sold_0[144] - Q0_var[144] - Q1_var[144]
   + total_stock_second_period_144 = 0
 R1447: total_stock_second_period_144 + Lost_1_expr_144
   = 348.6992635764772
 R1448: Sold_1[144] + Left_1[144] - total_stock_second_period_144 = 0
 R1449: - 700 Sold_0[144] - 700 Sold_1[144] + 700 Lost_0[144]
   + 700 Lost_1[144] + 300 Left_1[144] + profits_vars[144] = 0
 Fraction_lower_bound_145: Fraction[145] >= 0
 Fraction_upper_bound_145: Fraction[145] <= 1
 R1452: - 0.6922536662142053 alphas[0] + 0.4283477332294836 alphas[1]
   + 1.356922388533561 alphas[2] + 0.2041899883618861 alphas[3] - alphas[4]
   + f_var[145] = 0
 R1453: - 1857.770397056156 Fraction[145] + Q0_var[145] = 0
 R1454: Q0_var[145] + Lost_0_expr_145 = 338.5490396514093
 R1455: Q0_var[145] + Q_hat_adjusted_145 = 415.127733559304
 R1456: Sold_0[145] - Q0_var[145] - Q1_var[145]
   + total_stock_second_period_145 = 0
 R1457: total_stock_second_period_145 + Lost_1_expr_145
   = 76.53182423382945
 R1458: Sold_1[145] + Left_1[145] - total_stock_second_period_145 = 0
 R1459: - 700 Sold_0[145] - 700 Sold_1[145] + 700 Lost_0[145]
   + 700 Lost_1[145] + 300 Left_1[145] + profits_vars[145] = 0
 Fraction_lower_bound_146: Fraction[146] >= 0
 Fraction_upper_bound_146: Fraction[146] <= 1
 R1462: 1.466168824862132 alphas[0] - 1.153369686411517 alphas[1]
   - 1.429048366563702 alphas[2] - 0.5467278081497916 alphas[3] - alphas[4]
   + f_var[146] = 0
 R1463: - 1857.770397056156 Fraction[146] + Q0_var[146] = 0
 R1464: Q0_var[146] + Lost_0_expr_146 = 2137.115918935199
 R1465: Q0_var[146] + Q_hat_adjusted_146 = 2636.925883775706
 R1466: Sold_0[146] - Q0_var[146] - Q1_var[146]
   + total_stock_second_period_146 = 0
 R1467: total_stock_second_period_146 + Lost_1_expr_146
   = 499.7421172731004
 R1468: Sold_1[146] + Left_1[146] - total_stock_second_period_146 = 0
 R1469: - 700 Sold_0[146] - 700 Sold_1[146] + 700 Lost_0[146]
   + 700 Lost_1[146] + 300 Left_1[146] + profits_vars[146] = 0
 Fraction_lower_bound_147: Fraction[147] >= 0
 Fraction_upper_bound_147: Fraction[147] <= 1
 R1472: - 1.463120869885449 alphas[0] - 1.731618056816455 alphas[1]
   - 0.8642988875309594 alphas[2] - 1.226997029117752 alphas[3] - alphas[4]
   + f_var[147] = 0
 R1473: - 1857.770397056156 Fraction[147] + Q0_var[147] = 0
 R1474: Q0_var[147] + Lost_0_expr_147 = 1875.87637762776
 R1475: Q0_var[147] + Q_hat_adjusted_147 = 2339.741827191771
 R1476: Sold_0[147] - Q0_var[147] - Q1_var[147]
   + total_stock_second_period_147 = 0
 R1477: total_stock_second_period_147 + Lost_1_expr_147
   = 463.7923122432344
 R1478: Sold_1[147] + Left_1[147] - total_stock_second_period_147 = 0
 R1479: - 700 Sold_0[147] - 700 Sold_1[147] + 700 Lost_0[147]
   + 700 Lost_1[147] + 300 Left_1[147] + profits_vars[147] = 0
 Fraction_lower_bound_148: Fraction[148] >= 0
 Fraction_upper_bound_148: Fraction[148] <= 1
 R1482: - 0.7292703184494327 alphas[0] - 1.67559984373158 alphas[1]
   - 1.723100109918131 alphas[2] + 0.0512765910738639 alphas[3] - alphas[4]
   + f_var[148] = 0
 R1483: - 1857.770397056156 Fraction[148] + Q0_var[148] = 0
 R1484: Q0_var[148] + Lost_0_expr_148 = 2696.957938733008
 R1485: Q0_var[148] + Q_hat_adjusted_148 = 3289.125748205951
 R1486: Sold_0[148] - Q0_var[148] - Q1_var[148]
   + total_stock_second_period_148 = 0
 R1487: total_stock_second_period_148 + Lost_1_expr_148
   = 592.1020474743182
 R1488: Sold_1[148] + Left_1[148] - total_stock_second_period_148 = 0
 R1489: - 700 Sold_0[148] - 700 Sold_1[148] + 700 Lost_0[148]
   + 700 Lost_1[148] + 300 Left_1[148] + profits_vars[148] = 0
 Fraction_lower_bound_149: Fraction[149] >= 0
 Fraction_upper_bound_149: Fraction[149] <= 1
 R1492: - 1.746892213223223 alphas[0] - 1.458497282175861 alphas[1]
   - 0.1265561301846488 alphas[2] - 1.619217623960803 alphas[3] - alphas[4]
   + f_var[149] = 0
 R1493: - 1857.770397056156 Fraction[149] + Q0_var[149] = 0
 R1494: Q0_var[149] + Lost_0_expr_149 = 1641.775307931028
 R1495: Q0_var[149] + Q_hat_adjusted_149 = 1988.50672789382
 R1496: Sold_0[149] - Q0_var[149] - Q1_var[149]
   + total_stock_second_period_149 = 0
 R1497: total_stock_second_period_149 + Lost_1_expr_149 = 346.671016981829
 R1498: Sold_1[149] + Left_1[149] - total_stock_second_period_149 = 0
 R1499: - 700 Sold_0[149] - 700 Sold_1[149] + 700 Lost_0[149]
   + 700 Lost_1[149] + 300 Left_1[149] + profits_vars[149] = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
 alphas[4] free
 f_var[0] free
 f_var[1] free
 f_var[2] free
 f_var[3] free
 f_var[4] free
 f_var[5] free
 f_var[6] free
 f_var[7] free
 f_var[8] free
 f_var[9] free
 f_var[10] free
 f_var[11] free
 f_var[12] free
 f_var[13] free
 f_var[14] free
 f_var[15] free
 f_var[16] free
 f_var[17] free
 f_var[18] free
 f_var[19] free
 f_var[20] free
 f_var[21] free
 f_var[22] free
 f_var[23] free
 f_var[24] free
 f_var[25] free
 f_var[26] free
 f_var[27] free
 f_var[28] free
 f_var[29] free
 f_var[30] free
 f_var[31] free
 f_var[32] free
 f_var[33] free
 f_var[34] free
 f_var[35] free
 f_var[36] free
 f_var[37] free
 f_var[38] free
 f_var[39] free
 f_var[40] free
 f_var[41] free
 f_var[42] free
 f_var[43] free
 f_var[44] free
 f_var[45] free
 f_var[46] free
 f_var[47] free
 f_var[48] free
 f_var[49] free
 f_var[50] free
 f_var[51] free
 f_var[52] free
 f_var[53] free
 f_var[54] free
 f_var[55] free
 f_var[56] free
 f_var[57] free
 f_var[58] free
 f_var[59] free
 f_var[60] free
 f_var[61] free
 f_var[62] free
 f_var[63] free
 f_var[64] free
 f_var[65] free
 f_var[66] free
 f_var[67] free
 f_var[68] free
 f_var[69] free
 f_var[70] free
 f_var[71] free
 f_var[72] free
 f_var[73] free
 f_var[74] free
 f_var[75] free
 f_var[76] free
 f_var[77] free
 f_var[78] free
 f_var[79] free
 f_var[80] free
 f_var[81] free
 f_var[82] free
 f_var[83] free
 f_var[84] free
 f_var[85] free
 f_var[86] free
 f_var[87] free
 f_var[88] free
 f_var[89] free
 f_var[90] free
 f_var[91] free
 f_var[92] free
 f_var[93] free
 f_var[94] free
 f_var[95] free
 f_var[96] free
 f_var[97] free
 f_var[98] free
 f_var[99] free
 f_var[100] free
 f_var[101] free
 f_var[102] free
 f_var[103] free
 f_var[104] free
 f_var[105] free
 f_var[106] free
 f_var[107] free
 f_var[108] free
 f_var[109] free
 f_var[110] free
 f_var[111] free
 f_var[112] free
 f_var[113] free
 f_var[114] free
 f_var[115] free
 f_var[116] free
 f_var[117] free
 f_var[118] free
 f_var[119] free
 f_var[120] free
 f_var[121] free
 f_var[122] free
 f_var[123] free
 f_var[124] free
 f_var[125] free
 f_var[126] free
 f_var[127] free
 f_var[128] free
 f_var[129] free
 f_var[130] free
 f_var[131] free
 f_var[132] free
 f_var[133] free
 f_var[134] free
 f_var[135] free
 f_var[136] free
 f_var[137] free
 f_var[138] free
 f_var[139] free
 f_var[140] free
 f_var[141] free
 f_var[142] free
 f_var[143] free
 f_var[144] free
 f_var[145] free
 f_var[146] free
 f_var[147] free
 f_var[148] free
 f_var[149] free
 Fraction[0] <= 1
 Fraction[1] <= 1
 Fraction[2] <= 1
 Fraction[3] <= 1
 Fraction[4] <= 1
 Fraction[5] <= 1
 Fraction[6] <= 1
 Fraction[7] <= 1
 Fraction[8] <= 1
 Fraction[9] <= 1
 Fraction[10] <= 1
 Fraction[11] <= 1
 Fraction[12] <= 1
 Fraction[13] <= 1
 Fraction[14] <= 1
 Fraction[15] <= 1
 Fraction[16] <= 1
 Fraction[17] <= 1
 Fraction[18] <= 1
 Fraction[19] <= 1
 Fraction[20] <= 1
 Fraction[21] <= 1
 Fraction[22] <= 1
 Fraction[23] <= 1
 Fraction[24] <= 1
 Fraction[25] <= 1
 Fraction[26] <= 1
 Fraction[27] <= 1
 Fraction[28] <= 1
 Fraction[29] <= 1
 Fraction[30] <= 1
 Fraction[31] <= 1
 Fraction[32] <= 1
 Fraction[33] <= 1
 Fraction[34] <= 1
 Fraction[35] <= 1
 Fraction[36] <= 1
 Fraction[37] <= 1
 Fraction[38] <= 1
 Fraction[39] <= 1
 Fraction[40] <= 1
 Fraction[41] <= 1
 Fraction[42] <= 1
 Fraction[43] <= 1
 Fraction[44] <= 1
 Fraction[45] <= 1
 Fraction[46] <= 1
 Fraction[47] <= 1
 Fraction[48] <= 1
 Fraction[49] <= 1
 Fraction[50] <= 1
 Fraction[51] <= 1
 Fraction[52] <= 1
 Fraction[53] <= 1
 Fraction[54] <= 1
 Fraction[55] <= 1
 Fraction[56] <= 1
 Fraction[57] <= 1
 Fraction[58] <= 1
 Fraction[59] <= 1
 Fraction[60] <= 1
 Fraction[61] <= 1
 Fraction[62] <= 1
 Fraction[63] <= 1
 Fraction[64] <= 1
 Fraction[65] <= 1
 Fraction[66] <= 1
 Fraction[67] <= 1
 Fraction[68] <= 1
 Fraction[69] <= 1
 Fraction[70] <= 1
 Fraction[71] <= 1
 Fraction[72] <= 1
 Fraction[73] <= 1
 Fraction[74] <= 1
 Fraction[75] <= 1
 Fraction[76] <= 1
 Fraction[77] <= 1
 Fraction[78] <= 1
 Fraction[79] <= 1
 Fraction[80] <= 1
 Fraction[81] <= 1
 Fraction[82] <= 1
 Fraction[83] <= 1
 Fraction[84] <= 1
 Fraction[85] <= 1
 Fraction[86] <= 1
 Fraction[87] <= 1
 Fraction[88] <= 1
 Fraction[89] <= 1
 Fraction[90] <= 1
 Fraction[91] <= 1
 Fraction[92] <= 1
 Fraction[93] <= 1
 Fraction[94] <= 1
 Fraction[95] <= 1
 Fraction[96] <= 1
 Fraction[97] <= 1
 Fraction[98] <= 1
 Fraction[99] <= 1
 Fraction[100] <= 1
 Fraction[101] <= 1
 Fraction[102] <= 1
 Fraction[103] <= 1
 Fraction[104] <= 1
 Fraction[105] <= 1
 Fraction[106] <= 1
 Fraction[107] <= 1
 Fraction[108] <= 1
 Fraction[109] <= 1
 Fraction[110] <= 1
 Fraction[111] <= 1
 Fraction[112] <= 1
 Fraction[113] <= 1
 Fraction[114] <= 1
 Fraction[115] <= 1
 Fraction[116] <= 1
 Fraction[117] <= 1
 Fraction[118] <= 1
 Fraction[119] <= 1
 Fraction[120] <= 1
 Fraction[121] <= 1
 Fraction[122] <= 1
 Fraction[123] <= 1
 Fraction[124] <= 1
 Fraction[125] <= 1
 Fraction[126] <= 1
 Fraction[127] <= 1
 Fraction[128] <= 1
 Fraction[129] <= 1
 Fraction[130] <= 1
 Fraction[131] <= 1
 Fraction[132] <= 1
 Fraction[133] <= 1
 Fraction[134] <= 1
 Fraction[135] <= 1
 Fraction[136] <= 1
 Fraction[137] <= 1
 Fraction[138] <= 1
 Fraction[139] <= 1
 Fraction[140] <= 1
 Fraction[141] <= 1
 Fraction[142] <= 1
 Fraction[143] <= 1
 Fraction[144] <= 1
 Fraction[145] <= 1
 Fraction[146] <= 1
 Fraction[147] <= 1
 Fraction[148] <= 1
 Fraction[149] <= 1
 Q0_var[0] <= 1858.770397056156
 Q0_var[1] <= 1858.770397056156
 Q0_var[2] <= 1858.770397056156
 Q0_var[3] <= 1858.770397056156
 Q0_var[4] <= 1858.770397056156
 Q0_var[5] <= 1858.770397056156
 Q0_var[6] <= 1858.770397056156
 Q0_var[7] <= 1858.770397056156
 Q0_var[8] <= 1858.770397056156
 Q0_var[9] <= 1858.770397056156
 Q0_var[10] <= 1858.770397056156
 Q0_var[11] <= 1858.770397056156
 Q0_var[12] <= 1858.770397056156
 Q0_var[13] <= 1858.770397056156
 Q0_var[14] <= 1858.770397056156
 Q0_var[15] <= 1858.770397056156
 Q0_var[16] <= 1858.770397056156
 Q0_var[17] <= 1858.770397056156
 Q0_var[18] <= 1858.770397056156
 Q0_var[19] <= 1858.770397056156
 Q0_var[20] <= 1858.770397056156
 Q0_var[21] <= 1858.770397056156
 Q0_var[22] <= 1858.770397056156
 Q0_var[23] <= 1858.770397056156
 Q0_var[24] <= 1858.770397056156
 Q0_var[25] <= 1858.770397056156
 Q0_var[26] <= 1858.770397056156
 Q0_var[27] <= 1858.770397056156
 Q0_var[28] <= 1858.770397056156
 Q0_var[29] <= 1858.770397056156
 Q0_var[30] <= 1858.770397056156
 Q0_var[31] <= 1858.770397056156
 Q0_var[32] <= 1858.770397056156
 Q0_var[33] <= 1858.770397056156
 Q0_var[34] <= 1858.770397056156
 Q0_var[35] <= 1858.770397056156
 Q0_var[36] <= 1858.770397056156
 Q0_var[37] <= 1858.770397056156
 Q0_var[38] <= 1858.770397056156
 Q0_var[39] <= 1858.770397056156
 Q0_var[40] <= 1858.770397056156
 Q0_var[41] <= 1858.770397056156
 Q0_var[42] <= 1858.770397056156
 Q0_var[43] <= 1858.770397056156
 Q0_var[44] <= 1858.770397056156
 Q0_var[45] <= 1858.770397056156
 Q0_var[46] <= 1858.770397056156
 Q0_var[47] <= 1858.770397056156
 Q0_var[48] <= 1858.770397056156
 Q0_var[49] <= 1858.770397056156
 Q0_var[50] <= 1858.770397056156
 Q0_var[51] <= 1858.770397056156
 Q0_var[52] <= 1858.770397056156
 Q0_var[53] <= 1858.770397056156
 Q0_var[54] <= 1858.770397056156
 Q0_var[55] <= 1858.770397056156
 Q0_var[56] <= 1858.770397056156
 Q0_var[57] <= 1858.770397056156
 Q0_var[58] <= 1858.770397056156
 Q0_var[59] <= 1858.770397056156
 Q0_var[60] <= 1858.770397056156
 Q0_var[61] <= 1858.770397056156
 Q0_var[62] <= 1858.770397056156
 Q0_var[63] <= 1858.770397056156
 Q0_var[64] <= 1858.770397056156
 Q0_var[65] <= 1858.770397056156
 Q0_var[66] <= 1858.770397056156
 Q0_var[67] <= 1858.770397056156
 Q0_var[68] <= 1858.770397056156
 Q0_var[69] <= 1858.770397056156
 Q0_var[70] <= 1858.770397056156
 Q0_var[71] <= 1858.770397056156
 Q0_var[72] <= 1858.770397056156
 Q0_var[73] <= 1858.770397056156
 Q0_var[74] <= 1858.770397056156
 Q0_var[75] <= 1858.770397056156
 Q0_var[76] <= 1858.770397056156
 Q0_var[77] <= 1858.770397056156
 Q0_var[78] <= 1858.770397056156
 Q0_var[79] <= 1858.770397056156
 Q0_var[80] <= 1858.770397056156
 Q0_var[81] <= 1858.770397056156
 Q0_var[82] <= 1858.770397056156
 Q0_var[83] <= 1858.770397056156
 Q0_var[84] <= 1858.770397056156
 Q0_var[85] <= 1858.770397056156
 Q0_var[86] <= 1858.770397056156
 Q0_var[87] <= 1858.770397056156
 Q0_var[88] <= 1858.770397056156
 Q0_var[89] <= 1858.770397056156
 Q0_var[90] <= 1858.770397056156
 Q0_var[91] <= 1858.770397056156
 Q0_var[92] <= 1858.770397056156
 Q0_var[93] <= 1858.770397056156
 Q0_var[94] <= 1858.770397056156
 Q0_var[95] <= 1858.770397056156
 Q0_var[96] <= 1858.770397056156
 Q0_var[97] <= 1858.770397056156
 Q0_var[98] <= 1858.770397056156
 Q0_var[99] <= 1858.770397056156
 Q0_var[100] <= 1858.770397056156
 Q0_var[101] <= 1858.770397056156
 Q0_var[102] <= 1858.770397056156
 Q0_var[103] <= 1858.770397056156
 Q0_var[104] <= 1858.770397056156
 Q0_var[105] <= 1858.770397056156
 Q0_var[106] <= 1858.770397056156
 Q0_var[107] <= 1858.770397056156
 Q0_var[108] <= 1858.770397056156
 Q0_var[109] <= 1858.770397056156
 Q0_var[110] <= 1858.770397056156
 Q0_var[111] <= 1858.770397056156
 Q0_var[112] <= 1858.770397056156
 Q0_var[113] <= 1858.770397056156
 Q0_var[114] <= 1858.770397056156
 Q0_var[115] <= 1858.770397056156
 Q0_var[116] <= 1858.770397056156
 Q0_var[117] <= 1858.770397056156
 Q0_var[118] <= 1858.770397056156
 Q0_var[119] <= 1858.770397056156
 Q0_var[120] <= 1858.770397056156
 Q0_var[121] <= 1858.770397056156
 Q0_var[122] <= 1858.770397056156
 Q0_var[123] <= 1858.770397056156
 Q0_var[124] <= 1858.770397056156
 Q0_var[125] <= 1858.770397056156
 Q0_var[126] <= 1858.770397056156
 Q0_var[127] <= 1858.770397056156
 Q0_var[128] <= 1858.770397056156
 Q0_var[129] <= 1858.770397056156
 Q0_var[130] <= 1858.770397056156
 Q0_var[131] <= 1858.770397056156
 Q0_var[132] <= 1858.770397056156
 Q0_var[133] <= 1858.770397056156
 Q0_var[134] <= 1858.770397056156
 Q0_var[135] <= 1858.770397056156
 Q0_var[136] <= 1858.770397056156
 Q0_var[137] <= 1858.770397056156
 Q0_var[138] <= 1858.770397056156
 Q0_var[139] <= 1858.770397056156
 Q0_var[140] <= 1858.770397056156
 Q0_var[141] <= 1858.770397056156
 Q0_var[142] <= 1858.770397056156
 Q0_var[143] <= 1858.770397056156
 Q0_var[144] <= 1858.770397056156
 Q0_var[145] <= 1858.770397056156
 Q0_var[146] <= 1858.770397056156
 Q0_var[147] <= 1858.770397056156
 Q0_var[148] <= 1858.770397056156
 Q0_var[149] <= 1858.770397056156
 profits_vars[0] free
 profits_vars[1] free
 profits_vars[2] free
 profits_vars[3] free
 profits_vars[4] free
 profits_vars[5] free
 profits_vars[6] free
 profits_vars[7] free
 profits_vars[8] free
 profits_vars[9] free
 profits_vars[10] free
 profits_vars[11] free
 profits_vars[12] free
 profits_vars[13] free
 profits_vars[14] free
 profits_vars[15] free
 profits_vars[16] free
 profits_vars[17] free
 profits_vars[18] free
 profits_vars[19] free
 profits_vars[20] free
 profits_vars[21] free
 profits_vars[22] free
 profits_vars[23] free
 profits_vars[24] free
 profits_vars[25] free
 profits_vars[26] free
 profits_vars[27] free
 profits_vars[28] free
 profits_vars[29] free
 profits_vars[30] free
 profits_vars[31] free
 profits_vars[32] free
 profits_vars[33] free
 profits_vars[34] free
 profits_vars[35] free
 profits_vars[36] free
 profits_vars[37] free
 profits_vars[38] free
 profits_vars[39] free
 profits_vars[40] free
 profits_vars[41] free
 profits_vars[42] free
 profits_vars[43] free
 profits_vars[44] free
 profits_vars[45] free
 profits_vars[46] free
 profits_vars[47] free
 profits_vars[48] free
 profits_vars[49] free
 profits_vars[50] free
 profits_vars[51] free
 profits_vars[52] free
 profits_vars[53] free
 profits_vars[54] free
 profits_vars[55] free
 profits_vars[56] free
 profits_vars[57] free
 profits_vars[58] free
 profits_vars[59] free
 profits_vars[60] free
 profits_vars[61] free
 profits_vars[62] free
 profits_vars[63] free
 profits_vars[64] free
 profits_vars[65] free
 profits_vars[66] free
 profits_vars[67] free
 profits_vars[68] free
 profits_vars[69] free
 profits_vars[70] free
 profits_vars[71] free
 profits_vars[72] free
 profits_vars[73] free
 profits_vars[74] free
 profits_vars[75] free
 profits_vars[76] free
 profits_vars[77] free
 profits_vars[78] free
 profits_vars[79] free
 profits_vars[80] free
 profits_vars[81] free
 profits_vars[82] free
 profits_vars[83] free
 profits_vars[84] free
 profits_vars[85] free
 profits_vars[86] free
 profits_vars[87] free
 profits_vars[88] free
 profits_vars[89] free
 profits_vars[90] free
 profits_vars[91] free
 profits_vars[92] free
 profits_vars[93] free
 profits_vars[94] free
 profits_vars[95] free
 profits_vars[96] free
 profits_vars[97] free
 profits_vars[98] free
 profits_vars[99] free
 profits_vars[100] free
 profits_vars[101] free
 profits_vars[102] free
 profits_vars[103] free
 profits_vars[104] free
 profits_vars[105] free
 profits_vars[106] free
 profits_vars[107] free
 profits_vars[108] free
 profits_vars[109] free
 profits_vars[110] free
 profits_vars[111] free
 profits_vars[112] free
 profits_vars[113] free
 profits_vars[114] free
 profits_vars[115] free
 profits_vars[116] free
 profits_vars[117] free
 profits_vars[118] free
 profits_vars[119] free
 profits_vars[120] free
 profits_vars[121] free
 profits_vars[122] free
 profits_vars[123] free
 profits_vars[124] free
 profits_vars[125] free
 profits_vars[126] free
 profits_vars[127] free
 profits_vars[128] free
 profits_vars[129] free
 profits_vars[130] free
 profits_vars[131] free
 profits_vars[132] free
 profits_vars[133] free
 profits_vars[134] free
 profits_vars[135] free
 profits_vars[136] free
 profits_vars[137] free
 profits_vars[138] free
 profits_vars[139] free
 profits_vars[140] free
 profits_vars[141] free
 profits_vars[142] free
 profits_vars[143] free
 profits_vars[144] free
 profits_vars[145] free
 profits_vars[146] free
 profits_vars[147] free
 profits_vars[148] free
 profits_vars[149] free
 Lost_0_expr_0 free
 Q_hat_adjusted_0 free
 Lost_1_expr_0 free
 Lost_0_expr_1 free
 Q_hat_adjusted_1 free
 Lost_1_expr_1 free
 Lost_0_expr_2 free
 Q_hat_adjusted_2 free
 Lost_1_expr_2 free
 Lost_0_expr_3 free
 Q_hat_adjusted_3 free
 Lost_1_expr_3 free
 Lost_0_expr_4 free
 Q_hat_adjusted_4 free
 Lost_1_expr_4 free
 Lost_0_expr_5 free
 Q_hat_adjusted_5 free
 Lost_1_expr_5 free
 Lost_0_expr_6 free
 Q_hat_adjusted_6 free
 Lost_1_expr_6 free
 Lost_0_expr_7 free
 Q_hat_adjusted_7 free
 Lost_1_expr_7 free
 Lost_0_expr_8 free
 Q_hat_adjusted_8 free
 Lost_1_expr_8 free
 Lost_0_expr_9 free
 Q_hat_adjusted_9 free
 Lost_1_expr_9 free
 Lost_0_expr_10 free
 Q_hat_adjusted_10 free
 Lost_1_expr_10 free
 Lost_0_expr_11 free
 Q_hat_adjusted_11 free
 Lost_1_expr_11 free
 Lost_0_expr_12 free
 Q_hat_adjusted_12 free
 Lost_1_expr_12 free
 Lost_0_expr_13 free
 Q_hat_adjusted_13 free
 Lost_1_expr_13 free
 Lost_0_expr_14 free
 Q_hat_adjusted_14 free
 Lost_1_expr_14 free
 Lost_0_expr_15 free
 Q_hat_adjusted_15 free
 Lost_1_expr_15 free
 Lost_0_expr_16 free
 Q_hat_adjusted_16 free
 Lost_1_expr_16 free
 Lost_0_expr_17 free
 Q_hat_adjusted_17 free
 Lost_1_expr_17 free
 Lost_0_expr_18 free
 Q_hat_adjusted_18 free
 Lost_1_expr_18 free
 Lost_0_expr_19 free
 Q_hat_adjusted_19 free
 Lost_1_expr_19 free
 Lost_0_expr_20 free
 Q_hat_adjusted_20 free
 Lost_1_expr_20 free
 Lost_0_expr_21 free
 Q_hat_adjusted_21 free
 Lost_1_expr_21 free
 Lost_0_expr_22 free
 Q_hat_adjusted_22 free
 Lost_1_expr_22 free
 Lost_0_expr_23 free
 Q_hat_adjusted_23 free
 Lost_1_expr_23 free
 Lost_0_expr_24 free
 Q_hat_adjusted_24 free
 Lost_1_expr_24 free
 Lost_0_expr_25 free
 Q_hat_adjusted_25 free
 Lost_1_expr_25 free
 Lost_0_expr_26 free
 Q_hat_adjusted_26 free
 Lost_1_expr_26 free
 Lost_0_expr_27 free
 Q_hat_adjusted_27 free
 Lost_1_expr_27 free
 Lost_0_expr_28 free
 Q_hat_adjusted_28 free
 Lost_1_expr_28 free
 Lost_0_expr_29 free
 Q_hat_adjusted_29 free
 Lost_1_expr_29 free
 Lost_0_expr_30 free
 Q_hat_adjusted_30 free
 Lost_1_expr_30 free
 Lost_0_expr_31 free
 Q_hat_adjusted_31 free
 Lost_1_expr_31 free
 Lost_0_expr_32 free
 Q_hat_adjusted_32 free
 Lost_1_expr_32 free
 Lost_0_expr_33 free
 Q_hat_adjusted_33 free
 Lost_1_expr_33 free
 Lost_0_expr_34 free
 Q_hat_adjusted_34 free
 Lost_1_expr_34 free
 Lost_0_expr_35 free
 Q_hat_adjusted_35 free
 Lost_1_expr_35 free
 Lost_0_expr_36 free
 Q_hat_adjusted_36 free
 Lost_1_expr_36 free
 Lost_0_expr_37 free
 Q_hat_adjusted_37 free
 Lost_1_expr_37 free
 Lost_0_expr_38 free
 Q_hat_adjusted_38 free
 Lost_1_expr_38 free
 Lost_0_expr_39 free
 Q_hat_adjusted_39 free
 Lost_1_expr_39 free
 Lost_0_expr_40 free
 Q_hat_adjusted_40 free
 Lost_1_expr_40 free
 Lost_0_expr_41 free
 Q_hat_adjusted_41 free
 Lost_1_expr_41 free
 Lost_0_expr_42 free
 Q_hat_adjusted_42 free
 Lost_1_expr_42 free
 Lost_0_expr_43 free
 Q_hat_adjusted_43 free
 Lost_1_expr_43 free
 Lost_0_expr_44 free
 Q_hat_adjusted_44 free
 Lost_1_expr_44 free
 Lost_0_expr_45 free
 Q_hat_adjusted_45 free
 Lost_1_expr_45 free
 Lost_0_expr_46 free
 Q_hat_adjusted_46 free
 Lost_1_expr_46 free
 Lost_0_expr_47 free
 Q_hat_adjusted_47 free
 Lost_1_expr_47 free
 Lost_0_expr_48 free
 Q_hat_adjusted_48 free
 Lost_1_expr_48 free
 Lost_0_expr_49 free
 Q_hat_adjusted_49 free
 Lost_1_expr_49 free
 Lost_0_expr_50 free
 Q_hat_adjusted_50 free
 Lost_1_expr_50 free
 Lost_0_expr_51 free
 Q_hat_adjusted_51 free
 Lost_1_expr_51 free
 Lost_0_expr_52 free
 Q_hat_adjusted_52 free
 Lost_1_expr_52 free
 Lost_0_expr_53 free
 Q_hat_adjusted_53 free
 Lost_1_expr_53 free
 Lost_0_expr_54 free
 Q_hat_adjusted_54 free
 Lost_1_expr_54 free
 Lost_0_expr_55 free
 Q_hat_adjusted_55 free
 Lost_1_expr_55 free
 Lost_0_expr_56 free
 Q_hat_adjusted_56 free
 Lost_1_expr_56 free
 Lost_0_expr_57 free
 Q_hat_adjusted_57 free
 Lost_1_expr_57 free
 Lost_0_expr_58 free
 Q_hat_adjusted_58 free
 Lost_1_expr_58 free
 Lost_0_expr_59 free
 Q_hat_adjusted_59 free
 Lost_1_expr_59 free
 Lost_0_expr_60 free
 Q_hat_adjusted_60 free
 Lost_1_expr_60 free
 Lost_0_expr_61 free
 Q_hat_adjusted_61 free
 Lost_1_expr_61 free
 Lost_0_expr_62 free
 Q_hat_adjusted_62 free
 Lost_1_expr_62 free
 Lost_0_expr_63 free
 Q_hat_adjusted_63 free
 Lost_1_expr_63 free
 Lost_0_expr_64 free
 Q_hat_adjusted_64 free
 Lost_1_expr_64 free
 Lost_0_expr_65 free
 Q_hat_adjusted_65 free
 Lost_1_expr_65 free
 Lost_0_expr_66 free
 Q_hat_adjusted_66 free
 Lost_1_expr_66 free
 Lost_0_expr_67 free
 Q_hat_adjusted_67 free
 Lost_1_expr_67 free
 Lost_0_expr_68 free
 Q_hat_adjusted_68 free
 Lost_1_expr_68 free
 Lost_0_expr_69 free
 Q_hat_adjusted_69 free
 Lost_1_expr_69 free
 Lost_0_expr_70 free
 Q_hat_adjusted_70 free
 Lost_1_expr_70 free
 Lost_0_expr_71 free
 Q_hat_adjusted_71 free
 Lost_1_expr_71 free
 Lost_0_expr_72 free
 Q_hat_adjusted_72 free
 Lost_1_expr_72 free
 Lost_0_expr_73 free
 Q_hat_adjusted_73 free
 Lost_1_expr_73 free
 Lost_0_expr_74 free
 Q_hat_adjusted_74 free
 Lost_1_expr_74 free
 Lost_0_expr_75 free
 Q_hat_adjusted_75 free
 Lost_1_expr_75 free
 Lost_0_expr_76 free
 Q_hat_adjusted_76 free
 Lost_1_expr_76 free
 Lost_0_expr_77 free
 Q_hat_adjusted_77 free
 Lost_1_expr_77 free
 Lost_0_expr_78 free
 Q_hat_adjusted_78 free
 Lost_1_expr_78 free
 Lost_0_expr_79 free
 Q_hat_adjusted_79 free
 Lost_1_expr_79 free
 Lost_0_expr_80 free
 Q_hat_adjusted_80 free
 Lost_1_expr_80 free
 Lost_0_expr_81 free
 Q_hat_adjusted_81 free
 Lost_1_expr_81 free
 Lost_0_expr_82 free
 Q_hat_adjusted_82 free
 Lost_1_expr_82 free
 Lost_0_expr_83 free
 Q_hat_adjusted_83 free
 Lost_1_expr_83 free
 Lost_0_expr_84 free
 Q_hat_adjusted_84 free
 Lost_1_expr_84 free
 Lost_0_expr_85 free
 Q_hat_adjusted_85 free
 Lost_1_expr_85 free
 Lost_0_expr_86 free
 Q_hat_adjusted_86 free
 Lost_1_expr_86 free
 Lost_0_expr_87 free
 Q_hat_adjusted_87 free
 Lost_1_expr_87 free
 Lost_0_expr_88 free
 Q_hat_adjusted_88 free
 Lost_1_expr_88 free
 Lost_0_expr_89 free
 Q_hat_adjusted_89 free
 Lost_1_expr_89 free
 Lost_0_expr_90 free
 Q_hat_adjusted_90 free
 Lost_1_expr_90 free
 Lost_0_expr_91 free
 Q_hat_adjusted_91 free
 Lost_1_expr_91 free
 Lost_0_expr_92 free
 Q_hat_adjusted_92 free
 Lost_1_expr_92 free
 Lost_0_expr_93 free
 Q_hat_adjusted_93 free
 Lost_1_expr_93 free
 Lost_0_expr_94 free
 Q_hat_adjusted_94 free
 Lost_1_expr_94 free
 Lost_0_expr_95 free
 Q_hat_adjusted_95 free
 Lost_1_expr_95 free
 Lost_0_expr_96 free
 Q_hat_adjusted_96 free
 Lost_1_expr_96 free
 Lost_0_expr_97 free
 Q_hat_adjusted_97 free
 Lost_1_expr_97 free
 Lost_0_expr_98 free
 Q_hat_adjusted_98 free
 Lost_1_expr_98 free
 Lost_0_expr_99 free
 Q_hat_adjusted_99 free
 Lost_1_expr_99 free
 Lost_0_expr_100 free
 Q_hat_adjusted_100 free
 Lost_1_expr_100 free
 Lost_0_expr_101 free
 Q_hat_adjusted_101 free
 Lost_1_expr_101 free
 Lost_0_expr_102 free
 Q_hat_adjusted_102 free
 Lost_1_expr_102 free
 Lost_0_expr_103 free
 Q_hat_adjusted_103 free
 Lost_1_expr_103 free
 Lost_0_expr_104 free
 Q_hat_adjusted_104 free
 Lost_1_expr_104 free
 Lost_0_expr_105 free
 Q_hat_adjusted_105 free
 Lost_1_expr_105 free
 Lost_0_expr_106 free
 Q_hat_adjusted_106 free
 Lost_1_expr_106 free
 Lost_0_expr_107 free
 Q_hat_adjusted_107 free
 Lost_1_expr_107 free
 Lost_0_expr_108 free
 Q_hat_adjusted_108 free
 Lost_1_expr_108 free
 Lost_0_expr_109 free
 Q_hat_adjusted_109 free
 Lost_1_expr_109 free
 Lost_0_expr_110 free
 Q_hat_adjusted_110 free
 Lost_1_expr_110 free
 Lost_0_expr_111 free
 Q_hat_adjusted_111 free
 Lost_1_expr_111 free
 Lost_0_expr_112 free
 Q_hat_adjusted_112 free
 Lost_1_expr_112 free
 Lost_0_expr_113 free
 Q_hat_adjusted_113 free
 Lost_1_expr_113 free
 Lost_0_expr_114 free
 Q_hat_adjusted_114 free
 Lost_1_expr_114 free
 Lost_0_expr_115 free
 Q_hat_adjusted_115 free
 Lost_1_expr_115 free
 Lost_0_expr_116 free
 Q_hat_adjusted_116 free
 Lost_1_expr_116 free
 Lost_0_expr_117 free
 Q_hat_adjusted_117 free
 Lost_1_expr_117 free
 Lost_0_expr_118 free
 Q_hat_adjusted_118 free
 Lost_1_expr_118 free
 Lost_0_expr_119 free
 Q_hat_adjusted_119 free
 Lost_1_expr_119 free
 Lost_0_expr_120 free
 Q_hat_adjusted_120 free
 Lost_1_expr_120 free
 Lost_0_expr_121 free
 Q_hat_adjusted_121 free
 Lost_1_expr_121 free
 Lost_0_expr_122 free
 Q_hat_adjusted_122 free
 Lost_1_expr_122 free
 Lost_0_expr_123 free
 Q_hat_adjusted_123 free
 Lost_1_expr_123 free
 Lost_0_expr_124 free
 Q_hat_adjusted_124 free
 Lost_1_expr_124 free
 Lost_0_expr_125 free
 Q_hat_adjusted_125 free
 Lost_1_expr_125 free
 Lost_0_expr_126 free
 Q_hat_adjusted_126 free
 Lost_1_expr_126 free
 Lost_0_expr_127 free
 Q_hat_adjusted_127 free
 Lost_1_expr_127 free
 Lost_0_expr_128 free
 Q_hat_adjusted_128 free
 Lost_1_expr_128 free
 Lost_0_expr_129 free
 Q_hat_adjusted_129 free
 Lost_1_expr_129 free
 Lost_0_expr_130 free
 Q_hat_adjusted_130 free
 Lost_1_expr_130 free
 Lost_0_expr_131 free
 Q_hat_adjusted_131 free
 Lost_1_expr_131 free
 Lost_0_expr_132 free
 Q_hat_adjusted_132 free
 Lost_1_expr_132 free
 Lost_0_expr_133 free
 Q_hat_adjusted_133 free
 Lost_1_expr_133 free
 Lost_0_expr_134 free
 Q_hat_adjusted_134 free
 Lost_1_expr_134 free
 Lost_0_expr_135 free
 Q_hat_adjusted_135 free
 Lost_1_expr_135 free
 Lost_0_expr_136 free
 Q_hat_adjusted_136 free
 Lost_1_expr_136 free
 Lost_0_expr_137 free
 Q_hat_adjusted_137 free
 Lost_1_expr_137 free
 Lost_0_expr_138 free
 Q_hat_adjusted_138 free
 Lost_1_expr_138 free
 Lost_0_expr_139 free
 Q_hat_adjusted_139 free
 Lost_1_expr_139 free
 Lost_0_expr_140 free
 Q_hat_adjusted_140 free
 Lost_1_expr_140 free
 Lost_0_expr_141 free
 Q_hat_adjusted_141 free
 Lost_1_expr_141 free
 Lost_0_expr_142 free
 Q_hat_adjusted_142 free
 Lost_1_expr_142 free
 Lost_0_expr_143 free
 Q_hat_adjusted_143 free
 Lost_1_expr_143 free
 Lost_0_expr_144 free
 Q_hat_adjusted_144 free
 Lost_1_expr_144 free
 Lost_0_expr_145 free
 Q_hat_adjusted_145 free
 Lost_1_expr_145 free
 Lost_0_expr_146 free
 Q_hat_adjusted_146 free
 Lost_1_expr_146 free
 Lost_0_expr_147 free
 Q_hat_adjusted_147 free
 Lost_1_expr_147 free
 Lost_0_expr_148 free
 Q_hat_adjusted_148 free
 Lost_1_expr_148 free
 Lost_0_expr_149 free
 Q_hat_adjusted_149 free
 Lost_1_expr_149 free
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 730.3707927279808 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 168.7086446709725 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 869.6845049936463 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 254.633373824361 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1329.298118406662 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 378.8778909639552 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 377.7611703831191 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 43.82977226896783 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1794.210237585196 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 317.4725421336088 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1030.824017135466 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 238.8590514752216 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 803.8809012446486 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 153.6709822174253 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 403.1630961049649 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 130.9836691493348 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1266.450319396651 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 358.9331194955274 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 752.9232712374273 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 179.1106038049684 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 1652.066324940119 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 299.4777637564143 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 1065.655092867825 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 188.3267912559151 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 1980.867431116381 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 461.3656491148446 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 539.8468829401479 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 93.71170758669381 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 449.3767629861672 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 80.42412628960201 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 737.3331366144699 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 165.0386537877427 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1132.129505673431 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 315.1440193232037 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1488.526064406169 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 399.1573861422436 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1288.670206187575 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 285.8000152534103 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 2345.395139225224 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 592.1756591175944 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 701.7271029196665 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 204.3116202512401 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1393.512650710201 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 370.8283656243854 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 1432.682604551746 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 307.5706109011828 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1891.806475401024 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 420.9672444883834 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 2031.702061734177 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 485.1859808270665 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 1943.65065166081 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 475.2298694714289 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 1916.176524625012 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 424.0597340783264 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 1690.394039235398 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 451.9175557268797 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 897.4119354793039 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 174.794673699044 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 262.1780804630767 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 110.4194879087977 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 271.0327822046679 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 10.58615204986268 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 1097.106798181648 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 227.9024900319866 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 875.9743933206253 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 210.9594381737755 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 1413.147746884571 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 346.4596423887937 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 1964.532943950368 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 444.7870483758768 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 2052.016358371682 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 547.63768789984 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 311.3260626884876 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 85.42107893767221 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 397.5014433502334 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 100.0734071328117 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 1881.617660557855 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 372.9723559444992 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 1618.412973186389 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 383.2290680533446 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 1863.590338352213 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 460.4162802920647 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 1938.474854887897 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 483.0231304209928 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 2653.538574913986 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 555.1207929398457 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 695.0864625406235 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 162.5087358697913 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 1240.946505298687 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 280.0782332900822 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 365.4752556834671 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 80.60482418747404 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 1673.535330517504 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 367.4714805323185 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 1406.264080400039 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 339.9899378798741 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 2012.645614215695 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 466.9165313952075 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 518.0576627311875 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 128.7752493999675 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
 GC300: Fraction[50] = LOGISTIC ( f_var[50] )
 min_sales_constr_50: Sold_0[50] = MIN ( Q0_var[50] , 1155.543890860777 )
 max_lost_constr_50: Lost_0[50] = MAX ( Lost_0_expr_50 , 0 )
 max_Q1_constr_50: Q1_var[50] = MAX ( Q_hat_adjusted_50 , 0 )
 min_sales2_constr_50: Sold_1[50] = MIN ( total_stock_second_period_50
   , 323.5320368272452 )
 max_lost2_constr_50: Lost_1[50] = MAX ( Lost_1_expr_50 , 0 )
 GC306: Fraction[51] = LOGISTIC ( f_var[51] )
 min_sales_constr_51: Sold_0[51] = MIN ( Q0_var[51] , 1203.586068617233 )
 max_lost_constr_51: Lost_0[51] = MAX ( Lost_0_expr_51 , 0 )
 max_Q1_constr_51: Q1_var[51] = MAX ( Q_hat_adjusted_51 , 0 )
 min_sales2_constr_51: Sold_1[51] = MIN ( total_stock_second_period_51
   , 256.4495662927921 )
 max_lost2_constr_51: Lost_1[51] = MAX ( Lost_1_expr_51 , 0 )
 GC312: Fraction[52] = LOGISTIC ( f_var[52] )
 min_sales_constr_52: Sold_0[52] = MIN ( Q0_var[52] , 1128.752483923819 )
 max_lost_constr_52: Lost_0[52] = MAX ( Lost_0_expr_52 , 0 )
 max_Q1_constr_52: Q1_var[52] = MAX ( Q_hat_adjusted_52 , 0 )
 min_sales2_constr_52: Sold_1[52] = MIN ( total_stock_second_period_52
   , 188.5412448682371 )
 max_lost2_constr_52: Lost_1[52] = MAX ( Lost_1_expr_52 , 0 )
 GC318: Fraction[53] = LOGISTIC ( f_var[53] )
 min_sales_constr_53: Sold_0[53] = MIN ( Q0_var[53] , 606.111457300907 )
 max_lost_constr_53: Lost_0[53] = MAX ( Lost_0_expr_53 , 0 )
 max_Q1_constr_53: Q1_var[53] = MAX ( Q_hat_adjusted_53 , 0 )
 min_sales2_constr_53: Sold_1[53] = MIN ( total_stock_second_period_53
   , 121.2366780078799 )
 max_lost2_constr_53: Lost_1[53] = MAX ( Lost_1_expr_53 , 0 )
 GC324: Fraction[54] = LOGISTIC ( f_var[54] )
 min_sales_constr_54: Sold_0[54] = MIN ( Q0_var[54] , 2153.158103040867 )
 max_lost_constr_54: Lost_0[54] = MAX ( Lost_0_expr_54 , 0 )
 max_Q1_constr_54: Q1_var[54] = MAX ( Q_hat_adjusted_54 , 0 )
 min_sales2_constr_54: Sold_1[54] = MIN ( total_stock_second_period_54
   , 478.3687726861248 )
 max_lost2_constr_54: Lost_1[54] = MAX ( Lost_1_expr_54 , 0 )
 GC330: Fraction[55] = LOGISTIC ( f_var[55] )
 min_sales_constr_55: Sold_0[55] = MIN ( Q0_var[55] , 255.1921841831005 )
 max_lost_constr_55: Lost_0[55] = MAX ( Lost_0_expr_55 , 0 )
 max_Q1_constr_55: Q1_var[55] = MAX ( Q_hat_adjusted_55 , 0 )
 min_sales2_constr_55: Sold_1[55] = MIN ( total_stock_second_period_55
   , 86.74051335627254 )
 max_lost2_constr_55: Lost_1[55] = MAX ( Lost_1_expr_55 , 0 )
 GC336: Fraction[56] = LOGISTIC ( f_var[56] )
 min_sales_constr_56: Sold_0[56] = MIN ( Q0_var[56] , 1120.695096417727 )
 max_lost_constr_56: Lost_0[56] = MAX ( Lost_0_expr_56 , 0 )
 max_Q1_constr_56: Q1_var[56] = MAX ( Q_hat_adjusted_56 , 0 )
 min_sales2_constr_56: Sold_1[56] = MIN ( total_stock_second_period_56
   , 258.6176776940822 )
 max_lost2_constr_56: Lost_1[56] = MAX ( Lost_1_expr_56 , 0 )
 GC342: Fraction[57] = LOGISTIC ( f_var[57] )
 min_sales_constr_57: Sold_0[57] = MIN ( Q0_var[57] , 445.5600231954294 )
 max_lost_constr_57: Lost_0[57] = MAX ( Lost_0_expr_57 , 0 )
 max_Q1_constr_57: Q1_var[57] = MAX ( Q_hat_adjusted_57 , 0 )
 min_sales2_constr_57: Sold_1[57] = MIN ( total_stock_second_period_57
   , 45.69869434935595 )
 max_lost2_constr_57: Lost_1[57] = MAX ( Lost_1_expr_57 , 0 )
 GC348: Fraction[58] = LOGISTIC ( f_var[58] )
 min_sales_constr_58: Sold_0[58] = MIN ( Q0_var[58] , 1294.599640439763 )
 max_lost_constr_58: Lost_0[58] = MAX ( Lost_0_expr_58 , 0 )
 max_Q1_constr_58: Q1_var[58] = MAX ( Q_hat_adjusted_58 , 0 )
 min_sales2_constr_58: Sold_1[58] = MIN ( total_stock_second_period_58
   , 243.2324475751845 )
 max_lost2_constr_58: Lost_1[58] = MAX ( Lost_1_expr_58 , 0 )
 GC354: Fraction[59] = LOGISTIC ( f_var[59] )
 min_sales_constr_59: Sold_0[59] = MIN ( Q0_var[59] , 2279.901858514709 )
 max_lost_constr_59: Lost_0[59] = MAX ( Lost_0_expr_59 , 0 )
 max_Q1_constr_59: Q1_var[59] = MAX ( Q_hat_adjusted_59 , 0 )
 min_sales2_constr_59: Sold_1[59] = MIN ( total_stock_second_period_59
   , 562.0227537355461 )
 max_lost2_constr_59: Lost_1[59] = MAX ( Lost_1_expr_59 , 0 )
 GC360: Fraction[60] = LOGISTIC ( f_var[60] )
 min_sales_constr_60: Sold_0[60] = MIN ( Q0_var[60] , 1089.128129659249 )
 max_lost_constr_60: Lost_0[60] = MAX ( Lost_0_expr_60 , 0 )
 max_Q1_constr_60: Q1_var[60] = MAX ( Q_hat_adjusted_60 , 0 )
 min_sales2_constr_60: Sold_1[60] = MIN ( total_stock_second_period_60
   , 214.9373091538704 )
 max_lost2_constr_60: Lost_1[60] = MAX ( Lost_1_expr_60 , 0 )
 GC366: Fraction[61] = LOGISTIC ( f_var[61] )
 min_sales_constr_61: Sold_0[61] = MIN ( Q0_var[61] , 2031.232422504364 )
 max_lost_constr_61: Lost_0[61] = MAX ( Lost_0_expr_61 , 0 )
 max_Q1_constr_61: Q1_var[61] = MAX ( Q_hat_adjusted_61 , 0 )
 min_sales2_constr_61: Sold_1[61] = MIN ( total_stock_second_period_61
   , 499.6135742042624 )
 max_lost2_constr_61: Lost_1[61] = MAX ( Lost_1_expr_61 , 0 )
 GC372: Fraction[62] = LOGISTIC ( f_var[62] )
 min_sales_constr_62: Sold_0[62] = MIN ( Q0_var[62] , 640.8201850228692 )
 max_lost_constr_62: Lost_0[62] = MAX ( Lost_0_expr_62 , 0 )
 max_Q1_constr_62: Q1_var[62] = MAX ( Q_hat_adjusted_62 , 0 )
 min_sales2_constr_62: Sold_1[62] = MIN ( total_stock_second_period_62
   , 128.7816338889592 )
 max_lost2_constr_62: Lost_1[62] = MAX ( Lost_1_expr_62 , 0 )
 GC378: Fraction[63] = LOGISTIC ( f_var[63] )
 min_sales_constr_63: Sold_0[63] = MIN ( Q0_var[63] , 607.7469194774691 )
 max_lost_constr_63: Lost_0[63] = MAX ( Lost_0_expr_63 , 0 )
 max_Q1_constr_63: Q1_var[63] = MAX ( Q_hat_adjusted_63 , 0 )
 min_sales2_constr_63: Sold_1[63] = MIN ( total_stock_second_period_63
   , 160.6977291265339 )
 max_lost2_constr_63: Lost_1[63] = MAX ( Lost_1_expr_63 , 0 )
 GC384: Fraction[64] = LOGISTIC ( f_var[64] )
 min_sales_constr_64: Sold_0[64] = MIN ( Q0_var[64] , 922.4950129198998 )
 max_lost_constr_64: Lost_0[64] = MAX ( Lost_0_expr_64 , 0 )
 max_Q1_constr_64: Q1_var[64] = MAX ( Q_hat_adjusted_64 , 0 )
 min_sales2_constr_64: Sold_1[64] = MIN ( total_stock_second_period_64
   , 285.7850890080398 )
 max_lost2_constr_64: Lost_1[64] = MAX ( Lost_1_expr_64 , 0 )
 GC390: Fraction[65] = LOGISTIC ( f_var[65] )
 min_sales_constr_65: Sold_0[65] = MIN ( Q0_var[65] , 1114.227159963983 )
 max_lost_constr_65: Lost_0[65] = MAX ( Lost_0_expr_65 , 0 )
 max_Q1_constr_65: Q1_var[65] = MAX ( Q_hat_adjusted_65 , 0 )
 min_sales2_constr_65: Sold_1[65] = MIN ( total_stock_second_period_65
   , 240.1013189266381 )
 max_lost2_constr_65: Lost_1[65] = MAX ( Lost_1_expr_65 , 0 )
 GC396: Fraction[66] = LOGISTIC ( f_var[66] )
 min_sales_constr_66: Sold_0[66] = MIN ( Q0_var[66] , 1721.491955556769 )
 max_lost_constr_66: Lost_0[66] = MAX ( Lost_0_expr_66 , 0 )
 max_Q1_constr_66: Q1_var[66] = MAX ( Q_hat_adjusted_66 , 0 )
 min_sales2_constr_66: Sold_1[66] = MIN ( total_stock_second_period_66
   , 419.2020117553433 )
 max_lost2_constr_66: Lost_1[66] = MAX ( Lost_1_expr_66 , 0 )
 GC402: Fraction[67] = LOGISTIC ( f_var[67] )
 min_sales_constr_67: Sold_0[67] = MIN ( Q0_var[67] , 1183.261882704097 )
 max_lost_constr_67: Lost_0[67] = MAX ( Lost_0_expr_67 , 0 )
 max_Q1_constr_67: Q1_var[67] = MAX ( Q_hat_adjusted_67 , 0 )
 min_sales2_constr_67: Sold_1[67] = MIN ( total_stock_second_period_67
   , 178.006844404126 )
 max_lost2_constr_67: Lost_1[67] = MAX ( Lost_1_expr_67 , 0 )
 GC408: Fraction[68] = LOGISTIC ( f_var[68] )
 min_sales_constr_68: Sold_0[68] = MIN ( Q0_var[68] , 1815.729948644613 )
 max_lost_constr_68: Lost_0[68] = MAX ( Lost_0_expr_68 , 0 )
 max_Q1_constr_68: Q1_var[68] = MAX ( Q_hat_adjusted_68 , 0 )
 min_sales2_constr_68: Sold_1[68] = MIN ( total_stock_second_period_68
   , 479.1930478028278 )
 max_lost2_constr_68: Lost_1[68] = MAX ( Lost_1_expr_68 , 0 )
 GC414: Fraction[69] = LOGISTIC ( f_var[69] )
 min_sales_constr_69: Sold_0[69] = MIN ( Q0_var[69] , 1646.182380735384 )
 max_lost_constr_69: Lost_0[69] = MAX ( Lost_0_expr_69 , 0 )
 max_Q1_constr_69: Q1_var[69] = MAX ( Q_hat_adjusted_69 , 0 )
 min_sales2_constr_69: Sold_1[69] = MIN ( total_stock_second_period_69
   , 354.147251345354 )
 max_lost2_constr_69: Lost_1[69] = MAX ( Lost_1_expr_69 , 0 )
 GC420: Fraction[70] = LOGISTIC ( f_var[70] )
 min_sales_constr_70: Sold_0[70] = MIN ( Q0_var[70] , 1393.718420667341 )
 max_lost_constr_70: Lost_0[70] = MAX ( Lost_0_expr_70 , 0 )
 max_Q1_constr_70: Q1_var[70] = MAX ( Q_hat_adjusted_70 , 0 )
 min_sales2_constr_70: Sold_1[70] = MIN ( total_stock_second_period_70
   , 276.27484325421 )
 max_lost2_constr_70: Lost_1[70] = MAX ( Lost_1_expr_70 , 0 )
 GC426: Fraction[71] = LOGISTIC ( f_var[71] )
 min_sales_constr_71: Sold_0[71] = MIN ( Q0_var[71] , 1100.421304225794 )
 max_lost_constr_71: Lost_0[71] = MAX ( Lost_0_expr_71 , 0 )
 max_Q1_constr_71: Q1_var[71] = MAX ( Q_hat_adjusted_71 , 0 )
 min_sales2_constr_71: Sold_1[71] = MIN ( total_stock_second_period_71
   , 249.9663810258077 )
 max_lost2_constr_71: Lost_1[71] = MAX ( Lost_1_expr_71 , 0 )
 GC432: Fraction[72] = LOGISTIC ( f_var[72] )
 min_sales_constr_72: Sold_0[72] = MIN ( Q0_var[72] , 1597.042907110813 )
 max_lost_constr_72: Lost_0[72] = MAX ( Lost_0_expr_72 , 0 )
 max_Q1_constr_72: Q1_var[72] = MAX ( Q_hat_adjusted_72 , 0 )
 min_sales2_constr_72: Sold_1[72] = MIN ( total_stock_second_period_72
   , 382.6550906991539 )
 max_lost2_constr_72: Lost_1[72] = MAX ( Lost_1_expr_72 , 0 )
 GC438: Fraction[73] = LOGISTIC ( f_var[73] )
 min_sales_constr_73: Sold_0[73] = MIN ( Q0_var[73] , 525.3242745557617 )
 max_lost_constr_73: Lost_0[73] = MAX ( Lost_0_expr_73 , 0 )
 max_Q1_constr_73: Q1_var[73] = MAX ( Q_hat_adjusted_73 , 0 )
 min_sales2_constr_73: Sold_1[73] = MIN ( total_stock_second_period_73
   , 98.36355005637778 )
 max_lost2_constr_73: Lost_1[73] = MAX ( Lost_1_expr_73 , 0 )
 GC444: Fraction[74] = LOGISTIC ( f_var[74] )
 min_sales_constr_74: Sold_0[74] = MIN ( Q0_var[74] , 1443.002268030613 )
 max_lost_constr_74: Lost_0[74] = MAX ( Lost_0_expr_74 , 0 )
 max_Q1_constr_74: Q1_var[74] = MAX ( Q_hat_adjusted_74 , 0 )
 min_sales2_constr_74: Sold_1[74] = MIN ( total_stock_second_period_74
   , 412.657798536112 )
 max_lost2_constr_74: Lost_1[74] = MAX ( Lost_1_expr_74 , 0 )
 GC450: Fraction[75] = LOGISTIC ( f_var[75] )
 min_sales_constr_75: Sold_0[75] = MIN ( Q0_var[75] , 1434.238648658598 )
 max_lost_constr_75: Lost_0[75] = MAX ( Lost_0_expr_75 , 0 )
 max_Q1_constr_75: Q1_var[75] = MAX ( Q_hat_adjusted_75 , 0 )
 min_sales2_constr_75: Sold_1[75] = MIN ( total_stock_second_period_75
   , 322.7261684746407 )
 max_lost2_constr_75: Lost_1[75] = MAX ( Lost_1_expr_75 , 0 )
 GC456: Fraction[76] = LOGISTIC ( f_var[76] )
 min_sales_constr_76: Sold_0[76] = MIN ( Q0_var[76] , 1617.105693926821 )
 max_lost_constr_76: Lost_0[76] = MAX ( Lost_0_expr_76 , 0 )
 max_Q1_constr_76: Q1_var[76] = MAX ( Q_hat_adjusted_76 , 0 )
 min_sales2_constr_76: Sold_1[76] = MIN ( total_stock_second_period_76
   , 431.727956220174 )
 max_lost2_constr_76: Lost_1[76] = MAX ( Lost_1_expr_76 , 0 )
 GC462: Fraction[77] = LOGISTIC ( f_var[77] )
 min_sales_constr_77: Sold_0[77] = MIN ( Q0_var[77] , 1675.024757495721 )
 max_lost_constr_77: Lost_0[77] = MAX ( Lost_0_expr_77 , 0 )
 max_Q1_constr_77: Q1_var[77] = MAX ( Q_hat_adjusted_77 , 0 )
 min_sales2_constr_77: Sold_1[77] = MIN ( total_stock_second_period_77
   , 359.8332301529997 )
 max_lost2_constr_77: Lost_1[77] = MAX ( Lost_1_expr_77 , 0 )
 GC468: Fraction[78] = LOGISTIC ( f_var[78] )
 min_sales_constr_78: Sold_0[78] = MIN ( Q0_var[78] , 454.1989492184744 )
 max_lost_constr_78: Lost_0[78] = MAX ( Lost_0_expr_78 , 0 )
 max_Q1_constr_78: Q1_var[78] = MAX ( Q_hat_adjusted_78 , 0 )
 min_sales2_constr_78: Sold_1[78] = MIN ( total_stock_second_period_78
   , 92.96767687391183 )
 max_lost2_constr_78: Lost_1[78] = MAX ( Lost_1_expr_78 , 0 )
 GC474: Fraction[79] = LOGISTIC ( f_var[79] )
 min_sales_constr_79: Sold_0[79] = MIN ( Q0_var[79] , 1237.572879628393 )
 max_lost_constr_79: Lost_0[79] = MAX ( Lost_0_expr_79 , 0 )
 max_Q1_constr_79: Q1_var[79] = MAX ( Q_hat_adjusted_79 , 0 )
 min_sales2_constr_79: Sold_1[79] = MIN ( total_stock_second_period_79
   , 237.6161965135755 )
 max_lost2_constr_79: Lost_1[79] = MAX ( Lost_1_expr_79 , 0 )
 GC480: Fraction[80] = LOGISTIC ( f_var[80] )
 min_sales_constr_80: Sold_0[80] = MIN ( Q0_var[80] , 1108.194004189659 )
 max_lost_constr_80: Lost_0[80] = MAX ( Lost_0_expr_80 , 0 )
 max_Q1_constr_80: Q1_var[80] = MAX ( Q_hat_adjusted_80 , 0 )
 min_sales2_constr_80: Sold_1[80] = MIN ( total_stock_second_period_80
   , 264.5378697298788 )
 max_lost2_constr_80: Lost_1[80] = MAX ( Lost_1_expr_80 , 0 )
 GC486: Fraction[81] = LOGISTIC ( f_var[81] )
 min_sales_constr_81: Sold_0[81] = MIN ( Q0_var[81] , 528.4894221338071 )
 max_lost_constr_81: Lost_0[81] = MAX ( Lost_0_expr_81 , 0 )
 max_Q1_constr_81: Q1_var[81] = MAX ( Q_hat_adjusted_81 , 0 )
 min_sales2_constr_81: Sold_1[81] = MIN ( total_stock_second_period_81
   , 182.8220714829393 )
 max_lost2_constr_81: Lost_1[81] = MAX ( Lost_1_expr_81 , 0 )
 GC492: Fraction[82] = LOGISTIC ( f_var[82] )
 min_sales_constr_82: Sold_0[82] = MIN ( Q0_var[82] , 546.8272854362976 )
 max_lost_constr_82: Lost_0[82] = MAX ( Lost_0_expr_82 , 0 )
 max_Q1_constr_82: Q1_var[82] = MAX ( Q_hat_adjusted_82 , 0 )
 min_sales2_constr_82: Sold_1[82] = MIN ( total_stock_second_period_82
   , 85.88213386236083 )
 max_lost2_constr_82: Lost_1[82] = MAX ( Lost_1_expr_82 , 0 )
 GC498: Fraction[83] = LOGISTIC ( f_var[83] )
 min_sales_constr_83: Sold_0[83] = MIN ( Q0_var[83] , 1703.68743400237 )
 max_lost_constr_83: Lost_0[83] = MAX ( Lost_0_expr_83 , 0 )
 max_Q1_constr_83: Q1_var[83] = MAX ( Q_hat_adjusted_83 , 0 )
 min_sales2_constr_83: Sold_1[83] = MIN ( total_stock_second_period_83
   , 426.1229907358042 )
 max_lost2_constr_83: Lost_1[83] = MAX ( Lost_1_expr_83 , 0 )
 GC504: Fraction[84] = LOGISTIC ( f_var[84] )
 min_sales_constr_84: Sold_0[84] = MIN ( Q0_var[84] , 508.7821821660282 )
 max_lost_constr_84: Lost_0[84] = MAX ( Lost_0_expr_84 , 0 )
 max_Q1_constr_84: Q1_var[84] = MAX ( Q_hat_adjusted_84 , 0 )
 min_sales2_constr_84: Sold_1[84] = MIN ( total_stock_second_period_84
   , 122.2301980049901 )
 max_lost2_constr_84: Lost_1[84] = MAX ( Lost_1_expr_84 , 0 )
 GC510: Fraction[85] = LOGISTIC ( f_var[85] )
 min_sales_constr_85: Sold_0[85] = MIN ( Q0_var[85] , 1696.698661340503 )
 max_lost_constr_85: Lost_0[85] = MAX ( Lost_0_expr_85 , 0 )
 max_Q1_constr_85: Q1_var[85] = MAX ( Q_hat_adjusted_85 , 0 )
 min_sales2_constr_85: Sold_1[85] = MIN ( total_stock_second_period_85
   , 386.2102326034618 )
 max_lost2_constr_85: Lost_1[85] = MAX ( Lost_1_expr_85 , 0 )
 GC516: Fraction[86] = LOGISTIC ( f_var[86] )
 min_sales_constr_86: Sold_0[86] = MIN ( Q0_var[86] , 1733.507241912456 )
 max_lost_constr_86: Lost_0[86] = MAX ( Lost_0_expr_86 , 0 )
 max_Q1_constr_86: Q1_var[86] = MAX ( Q_hat_adjusted_86 , 0 )
 min_sales2_constr_86: Sold_1[86] = MIN ( total_stock_second_period_86
   , 418.5220379925635 )
 max_lost2_constr_86: Lost_1[86] = MAX ( Lost_1_expr_86 , 0 )
 GC522: Fraction[87] = LOGISTIC ( f_var[87] )
 min_sales_constr_87: Sold_0[87] = MIN ( Q0_var[87] , 1261.496370150473 )
 max_lost_constr_87: Lost_0[87] = MAX ( Lost_0_expr_87 , 0 )
 max_Q1_constr_87: Q1_var[87] = MAX ( Q_hat_adjusted_87 , 0 )
 min_sales2_constr_87: Sold_1[87] = MIN ( total_stock_second_period_87
   , 289.9950482574426 )
 max_lost2_constr_87: Lost_1[87] = MAX ( Lost_1_expr_87 , 0 )
 GC528: Fraction[88] = LOGISTIC ( f_var[88] )
 min_sales_constr_88: Sold_0[88] = MIN ( Q0_var[88] , 2335.536163489294 )
 max_lost_constr_88: Lost_0[88] = MAX ( Lost_0_expr_88 , 0 )
 max_Q1_constr_88: Q1_var[88] = MAX ( Q_hat_adjusted_88 , 0 )
 min_sales2_constr_88: Sold_1[88] = MIN ( total_stock_second_period_88
   , 552.5971811280234 )
 max_lost2_constr_88: Lost_1[88] = MAX ( Lost_1_expr_88 , 0 )
 GC534: Fraction[89] = LOGISTIC ( f_var[89] )
 min_sales_constr_89: Sold_0[89] = MIN ( Q0_var[89] , 316.334619516777 )
 max_lost_constr_89: Lost_0[89] = MAX ( Lost_0_expr_89 , 0 )
 max_Q1_constr_89: Q1_var[89] = MAX ( Q_hat_adjusted_89 , 0 )
 min_sales2_constr_89: Sold_1[89] = MIN ( total_stock_second_period_89
   , 88.85781446285435 )
 max_lost2_constr_89: Lost_1[89] = MAX ( Lost_1_expr_89 , 0 )
 GC540: Fraction[90] = LOGISTIC ( f_var[90] )
 min_sales_constr_90: Sold_0[90] = MIN ( Q0_var[90] , 1780.119170907275 )
 max_lost_constr_90: Lost_0[90] = MAX ( Lost_0_expr_90 , 0 )
 max_Q1_constr_90: Q1_var[90] = MAX ( Q_hat_adjusted_90 , 0 )
 min_sales2_constr_90: Sold_1[90] = MIN ( total_stock_second_period_90
   , 436.4388421171717 )
 max_lost2_constr_90: Lost_1[90] = MAX ( Lost_1_expr_90 , 0 )
 GC546: Fraction[91] = LOGISTIC ( f_var[91] )
 min_sales_constr_91: Sold_0[91] = MIN ( Q0_var[91] , 377.6154961474439 )
 max_lost_constr_91: Lost_0[91] = MAX ( Lost_0_expr_91 , 0 )
 max_Q1_constr_91: Q1_var[91] = MAX ( Q_hat_adjusted_91 , 0 )
 min_sales2_constr_91: Sold_1[91] = MIN ( total_stock_second_period_91
   , 62.33601784942824 )
 max_lost2_constr_91: Lost_1[91] = MAX ( Lost_1_expr_91 , 0 )
 GC552: Fraction[92] = LOGISTIC ( f_var[92] )
 min_sales_constr_92: Sold_0[92] = MIN ( Q0_var[92] , 913.675091987402 )
 max_lost_constr_92: Lost_0[92] = MAX ( Lost_0_expr_92 , 0 )
 max_Q1_constr_92: Q1_var[92] = MAX ( Q_hat_adjusted_92 , 0 )
 min_sales2_constr_92: Sold_1[92] = MIN ( total_stock_second_period_92
   , 277.6635287002167 )
 max_lost2_constr_92: Lost_1[92] = MAX ( Lost_1_expr_92 , 0 )
 GC558: Fraction[93] = LOGISTIC ( f_var[93] )
 min_sales_constr_93: Sold_0[93] = MIN ( Q0_var[93] , 1178.211683657832 )
 max_lost_constr_93: Lost_0[93] = MAX ( Lost_0_expr_93 , 0 )
 max_Q1_constr_93: Q1_var[93] = MAX ( Q_hat_adjusted_93 , 0 )
 min_sales2_constr_93: Sold_1[93] = MIN ( total_stock_second_period_93
   , 295.4587723077492 )
 max_lost2_constr_93: Lost_1[93] = MAX ( Lost_1_expr_93 , 0 )
 GC564: Fraction[94] = LOGISTIC ( f_var[94] )
 min_sales_constr_94: Sold_0[94] = MIN ( Q0_var[94] , 1018.296817460222 )
 max_lost_constr_94: Lost_0[94] = MAX ( Lost_0_expr_94 , 0 )
 max_Q1_constr_94: Q1_var[94] = MAX ( Q_hat_adjusted_94 , 0 )
 min_sales2_constr_94: Sold_1[94] = MIN ( total_stock_second_period_94
   , 231.2643991345874 )
 max_lost2_constr_94: Lost_1[94] = MAX ( Lost_1_expr_94 , 0 )
 GC570: Fraction[95] = LOGISTIC ( f_var[95] )
 min_sales_constr_95: Sold_0[95] = MIN ( Q0_var[95] , 883.1668838513926 )
 max_lost_constr_95: Lost_0[95] = MAX ( Lost_0_expr_95 , 0 )
 max_Q1_constr_95: Q1_var[95] = MAX ( Q_hat_adjusted_95 , 0 )
 min_sales2_constr_95: Sold_1[95] = MIN ( total_stock_second_period_95
   , 284.1222614374472 )
 max_lost2_constr_95: Lost_1[95] = MAX ( Lost_1_expr_95 , 0 )
 GC576: Fraction[96] = LOGISTIC ( f_var[96] )
 min_sales_constr_96: Sold_0[96] = MIN ( Q0_var[96] , 556.6304201864218 )
 max_lost_constr_96: Lost_0[96] = MAX ( Lost_0_expr_96 , 0 )
 max_Q1_constr_96: Q1_var[96] = MAX ( Q_hat_adjusted_96 , 0 )
 min_sales2_constr_96: Sold_1[96] = MIN ( total_stock_second_period_96
   , 123.8651451030456 )
 max_lost2_constr_96: Lost_1[96] = MAX ( Lost_1_expr_96 , 0 )
 GC582: Fraction[97] = LOGISTIC ( f_var[97] )
 min_sales_constr_97: Sold_0[97] = MIN ( Q0_var[97] , 198.7501399662573 )
 max_lost_constr_97: Lost_0[97] = MAX ( Lost_0_expr_97 , 0 )
 max_Q1_constr_97: Q1_var[97] = MAX ( Q_hat_adjusted_97 , 0 )
 min_sales2_constr_97: Sold_1[97] = MIN ( total_stock_second_period_97
   , 1.923033295570896 )
 max_lost2_constr_97: Lost_1[97] = MAX ( Lost_1_expr_97 , 0 )
 GC588: Fraction[98] = LOGISTIC ( f_var[98] )
 min_sales_constr_98: Sold_0[98] = MIN ( Q0_var[98] , 1272.512911022482 )
 max_lost_constr_98: Lost_0[98] = MAX ( Lost_0_expr_98 , 0 )
 max_Q1_constr_98: Q1_var[98] = MAX ( Q_hat_adjusted_98 , 0 )
 min_sales2_constr_98: Sold_1[98] = MIN ( total_stock_second_period_98
   , 268.1198628484569 )
 max_lost2_constr_98: Lost_1[98] = MAX ( Lost_1_expr_98 , 0 )
 GC594: Fraction[99] = LOGISTIC ( f_var[99] )
 min_sales_constr_99: Sold_0[99] = MIN ( Q0_var[99] , 1804.404463805455 )
 max_lost_constr_99: Lost_0[99] = MAX ( Lost_0_expr_99 , 0 )
 max_Q1_constr_99: Q1_var[99] = MAX ( Q_hat_adjusted_99 , 0 )
 min_sales2_constr_99: Sold_1[99] = MIN ( total_stock_second_period_99
   , 399.7922801554564 )
 max_lost2_constr_99: Lost_1[99] = MAX ( Lost_1_expr_99 , 0 )
 GC600: Fraction[100] = LOGISTIC ( f_var[100] )
 min_sales_constr_100: Sold_0[100] = MIN ( Q0_var[100] , 979.4206476561085 )
 max_lost_constr_100: Lost_0[100] = MAX ( Lost_0_expr_100 , 0 )
 max_Q1_constr_100: Q1_var[100] = MAX ( Q_hat_adjusted_100 , 0 )
 min_sales2_constr_100: Sold_1[100] = MIN ( total_stock_second_period_100
   , 221.2303651108812 )
 max_lost2_constr_100: Lost_1[100] = MAX ( Lost_1_expr_100 , 0 )
 GC606: Fraction[101] = LOGISTIC ( f_var[101] )
 min_sales_constr_101: Sold_0[101] = MIN ( Q0_var[101] , 992.0590712696317 )
 max_lost_constr_101: Lost_0[101] = MAX ( Lost_0_expr_101 , 0 )
 max_Q1_constr_101: Q1_var[101] = MAX ( Q_hat_adjusted_101 , 0 )
 min_sales2_constr_101: Sold_1[101] = MIN ( total_stock_second_period_101
   , 216.4184466869513 )
 max_lost2_constr_101: Lost_1[101] = MAX ( Lost_1_expr_101 , 0 )
 GC612: Fraction[102] = LOGISTIC ( f_var[102] )
 min_sales_constr_102: Sold_0[102] = MIN ( Q0_var[102] , 353.5273035153854 )
 max_lost_constr_102: Lost_0[102] = MAX ( Lost_0_expr_102 , 0 )
 max_Q1_constr_102: Q1_var[102] = MAX ( Q_hat_adjusted_102 , 0 )
 min_sales2_constr_102: Sold_1[102] = MIN ( total_stock_second_period_102
   , 45.66830291453748 )
 max_lost2_constr_102: Lost_1[102] = MAX ( Lost_1_expr_102 , 0 )
 GC618: Fraction[103] = LOGISTIC ( f_var[103] )
 min_sales_constr_103: Sold_0[103] = MIN ( Q0_var[103] , 640.2924529199969 )
 max_lost_constr_103: Lost_0[103] = MAX ( Lost_0_expr_103 , 0 )
 max_Q1_constr_103: Q1_var[103] = MAX ( Q_hat_adjusted_103 , 0 )
 min_sales2_constr_103: Sold_1[103] = MIN ( total_stock_second_period_103
   , 149.3021682856962 )
 max_lost2_constr_103: Lost_1[103] = MAX ( Lost_1_expr_103 , 0 )
 GC624: Fraction[104] = LOGISTIC ( f_var[104] )
 min_sales_constr_104: Sold_0[104] = MIN ( Q0_var[104] , 877.9304765979887 )
 max_lost_constr_104: Lost_0[104] = MAX ( Lost_0_expr_104 , 0 )
 max_Q1_constr_104: Q1_var[104] = MAX ( Q_hat_adjusted_104 , 0 )
 min_sales2_constr_104: Sold_1[104] = MIN ( total_stock_second_period_104
   , 206.3017517732627 )
 max_lost2_constr_104: Lost_1[104] = MAX ( Lost_1_expr_104 , 0 )
 GC630: Fraction[105] = LOGISTIC ( f_var[105] )
 min_sales_constr_105: Sold_0[105] = MIN ( Q0_var[105] , 526.3605294604208 )
 max_lost_constr_105: Lost_0[105] = MAX ( Lost_0_expr_105 , 0 )
 max_Q1_constr_105: Q1_var[105] = MAX ( Q_hat_adjusted_105 , 0 )
 min_sales2_constr_105: Sold_1[105] = MIN ( total_stock_second_period_105
   , 75.44574864592298 )
 max_lost2_constr_105: Lost_1[105] = MAX ( Lost_1_expr_105 , 0 )
 GC636: Fraction[106] = LOGISTIC ( f_var[106] )
 min_sales_constr_106: Sold_0[106] = MIN ( Q0_var[106] , 712.4491036870108 )
 max_lost_constr_106: Lost_0[106] = MAX ( Lost_0_expr_106 , 0 )
 max_Q1_constr_106: Q1_var[106] = MAX ( Q_hat_adjusted_106 , 0 )
 min_sales2_constr_106: Sold_1[106] = MIN ( total_stock_second_period_106
   , 157.676172081327 )
 max_lost2_constr_106: Lost_1[106] = MAX ( Lost_1_expr_106 , 0 )
 GC642: Fraction[107] = LOGISTIC ( f_var[107] )
 min_sales_constr_107: Sold_0[107] = MIN ( Q0_var[107] , 487.614092931941 )
 max_lost_constr_107: Lost_0[107] = MAX ( Lost_0_expr_107 , 0 )
 max_Q1_constr_107: Q1_var[107] = MAX ( Q_hat_adjusted_107 , 0 )
 min_sales2_constr_107: Sold_1[107] = MIN ( total_stock_second_period_107
   , 65.74145737201637 )
 max_lost2_constr_107: Lost_1[107] = MAX ( Lost_1_expr_107 , 0 )
 GC648: Fraction[108] = LOGISTIC ( f_var[108] )
 min_sales_constr_108: Sold_0[108] = MIN ( Q0_var[108] , 320.6134520234763 )
 max_lost_constr_108: Lost_0[108] = MAX ( Lost_0_expr_108 , 0 )
 max_Q1_constr_108: Q1_var[108] = MAX ( Q_hat_adjusted_108 , 0 )
 min_sales2_constr_108: Sold_1[108] = MIN ( total_stock_second_period_108
   , 89.55871571389466 )
 max_lost2_constr_108: Lost_1[108] = MAX ( Lost_1_expr_108 , 0 )
 GC654: Fraction[109] = LOGISTIC ( f_var[109] )
 min_sales_constr_109: Sold_0[109] = MIN ( Q0_var[109] , 1946.685845362863 )
 max_lost_constr_109: Lost_0[109] = MAX ( Lost_0_expr_109 , 0 )
 max_Q1_constr_109: Q1_var[109] = MAX ( Q_hat_adjusted_109 , 0 )
 min_sales2_constr_109: Sold_1[109] = MIN ( total_stock_second_period_109
   , 415.1437698934867 )
 max_lost2_constr_109: Lost_1[109] = MAX ( Lost_1_expr_109 , 0 )
 GC660: Fraction[110] = LOGISTIC ( f_var[110] )
 min_sales_constr_110: Sold_0[110] = MIN ( Q0_var[110] , 222.1762921283521 )
 max_lost_constr_110: Lost_0[110] = MAX ( Lost_0_expr_110 , 0 )
 max_Q1_constr_110: Q1_var[110] = MAX ( Q_hat_adjusted_110 , 0 )
 min_sales2_constr_110: Sold_1[110] = MIN ( total_stock_second_period_110
   , 60.59888041210703 )
 max_lost2_constr_110: Lost_1[110] = MAX ( Lost_1_expr_110 , 0 )
 GC666: Fraction[111] = LOGISTIC ( f_var[111] )
 min_sales_constr_111: Sold_0[111] = MIN ( Q0_var[111] , 1121.570158585745 )
 max_lost_constr_111: Lost_0[111] = MAX ( Lost_0_expr_111 , 0 )
 max_Q1_constr_111: Q1_var[111] = MAX ( Q_hat_adjusted_111 , 0 )
 min_sales2_constr_111: Sold_1[111] = MIN ( total_stock_second_period_111
   , 343.9279303010536 )
 max_lost2_constr_111: Lost_1[111] = MAX ( Lost_1_expr_111 , 0 )
 GC672: Fraction[112] = LOGISTIC ( f_var[112] )
 min_sales_constr_112: Sold_0[112] = MIN ( Q0_var[112] , 1729.40645281362 )
 max_lost_constr_112: Lost_0[112] = MAX ( Lost_0_expr_112 , 0 )
 max_Q1_constr_112: Q1_var[112] = MAX ( Q_hat_adjusted_112 , 0 )
 min_sales2_constr_112: Sold_1[112] = MIN ( total_stock_second_period_112
   , 394.4478485244075 )
 max_lost2_constr_112: Lost_1[112] = MAX ( Lost_1_expr_112 , 0 )
 GC678: Fraction[113] = LOGISTIC ( f_var[113] )
 min_sales_constr_113: Sold_0[113] = MIN ( Q0_var[113] , 853.1071013674194 )
 max_lost_constr_113: Lost_0[113] = MAX ( Lost_0_expr_113 , 0 )
 max_Q1_constr_113: Q1_var[113] = MAX ( Q_hat_adjusted_113 , 0 )
 min_sales2_constr_113: Sold_1[113] = MIN ( total_stock_second_period_113
   , 235.1408307602986 )
 max_lost2_constr_113: Lost_1[113] = MAX ( Lost_1_expr_113 , 0 )
 GC684: Fraction[114] = LOGISTIC ( f_var[114] )
 min_sales_constr_114: Sold_0[114] = MIN ( Q0_var[114] , 503.1413172655861 )
 max_lost_constr_114: Lost_0[114] = MAX ( Lost_0_expr_114 , 0 )
 max_Q1_constr_114: Q1_var[114] = MAX ( Q_hat_adjusted_114 , 0 )
 min_sales2_constr_114: Sold_1[114] = MIN ( total_stock_second_period_114
   , 118.011455290041 )
 max_lost2_constr_114: Lost_1[114] = MAX ( Lost_1_expr_114 , 0 )
 GC690: Fraction[115] = LOGISTIC ( f_var[115] )
 min_sales_constr_115: Sold_0[115] = MIN ( Q0_var[115] , 614.6329623680242 )
 max_lost_constr_115: Lost_0[115] = MAX ( Lost_0_expr_115 , 0 )
 max_Q1_constr_115: Q1_var[115] = MAX ( Q_hat_adjusted_115 , 0 )
 min_sales2_constr_115: Sold_1[115] = MIN ( total_stock_second_period_115
   , 112.6504198975765 )
 max_lost2_constr_115: Lost_1[115] = MAX ( Lost_1_expr_115 , 0 )
 GC696: Fraction[116] = LOGISTIC ( f_var[116] )
 min_sales_constr_116: Sold_0[116] = MIN ( Q0_var[116] , 799.040100767827 )
 max_lost_constr_116: Lost_0[116] = MAX ( Lost_0_expr_116 , 0 )
 max_Q1_constr_116: Q1_var[116] = MAX ( Q_hat_adjusted_116 , 0 )
 min_sales2_constr_116: Sold_1[116] = MIN ( total_stock_second_period_116
   , 230.958309854928 )
 max_lost2_constr_116: Lost_1[116] = MAX ( Lost_1_expr_116 , 0 )
 GC702: Fraction[117] = LOGISTIC ( f_var[117] )
 min_sales_constr_117: Sold_0[117] = MIN ( Q0_var[117] , 985.5575272988551 )
 max_lost_constr_117: Lost_0[117] = MAX ( Lost_0_expr_117 , 0 )
 max_Q1_constr_117: Q1_var[117] = MAX ( Q_hat_adjusted_117 , 0 )
 min_sales2_constr_117: Sold_1[117] = MIN ( total_stock_second_period_117
   , 262.1389877326196 )
 max_lost2_constr_117: Lost_1[117] = MAX ( Lost_1_expr_117 , 0 )
 GC708: Fraction[118] = LOGISTIC ( f_var[118] )
 min_sales_constr_118: Sold_0[118] = MIN ( Q0_var[118] , 965.5760113542973 )
 max_lost_constr_118: Lost_0[118] = MAX ( Lost_0_expr_118 , 0 )
 max_Q1_constr_118: Q1_var[118] = MAX ( Q_hat_adjusted_118 , 0 )
 min_sales2_constr_118: Sold_1[118] = MIN ( total_stock_second_period_118
   , 183.37591264488 )
 max_lost2_constr_118: Lost_1[118] = MAX ( Lost_1_expr_118 , 0 )
 GC714: Fraction[119] = LOGISTIC ( f_var[119] )
 min_sales_constr_119: Sold_0[119] = MIN ( Q0_var[119] , 1758.06134228584 )
 max_lost_constr_119: Lost_0[119] = MAX ( Lost_0_expr_119 , 0 )
 max_Q1_constr_119: Q1_var[119] = MAX ( Q_hat_adjusted_119 , 0 )
 min_sales2_constr_119: Sold_1[119] = MIN ( total_stock_second_period_119
   , 445.6028865797322 )
 max_lost2_constr_119: Lost_1[119] = MAX ( Lost_1_expr_119 , 0 )
 GC720: Fraction[120] = LOGISTIC ( f_var[120] )
 min_sales_constr_120: Sold_0[120] = MIN ( Q0_var[120] , 1891.238243961441 )
 max_lost_constr_120: Lost_0[120] = MAX ( Lost_0_expr_120 , 0 )
 max_Q1_constr_120: Q1_var[120] = MAX ( Q_hat_adjusted_120 , 0 )
 min_sales2_constr_120: Sold_1[120] = MIN ( total_stock_second_period_120
   , 495.0721686963736 )
 max_lost2_constr_120: Lost_1[120] = MAX ( Lost_1_expr_120 , 0 )
 GC726: Fraction[121] = LOGISTIC ( f_var[121] )
 min_sales_constr_121: Sold_0[121] = MIN ( Q0_var[121] , 2295.92918895746 )
 max_lost_constr_121: Lost_0[121] = MAX ( Lost_0_expr_121 , 0 )
 max_Q1_constr_121: Q1_var[121] = MAX ( Q_hat_adjusted_121 , 0 )
 min_sales2_constr_121: Sold_1[121] = MIN ( total_stock_second_period_121
   , 560.7540796411829 )
 max_lost2_constr_121: Lost_1[121] = MAX ( Lost_1_expr_121 , 0 )
 GC732: Fraction[122] = LOGISTIC ( f_var[122] )
 min_sales_constr_122: Sold_0[122] = MIN ( Q0_var[122] , 125.6735446919541 )
 max_lost_constr_122: Lost_0[122] = MAX ( Lost_0_expr_122 , 0 )
 max_Q1_constr_122: Q1_var[122] = MAX ( Q_hat_adjusted_122 , 0 )
 min_sales2_constr_122: Sold_1[122] = MIN ( total_stock_second_period_122
   , 59.29501813212444 )
 max_lost2_constr_122: Lost_1[122] = MAX ( Lost_1_expr_122 , 0 )
 GC738: Fraction[123] = LOGISTIC ( f_var[123] )
 min_sales_constr_123: Sold_0[123] = MIN ( Q0_var[123] , 2042.760817796573 )
 max_lost_constr_123: Lost_0[123] = MAX ( Lost_0_expr_123 , 0 )
 max_Q1_constr_123: Q1_var[123] = MAX ( Q_hat_adjusted_123 , 0 )
 min_sales2_constr_123: Sold_1[123] = MIN ( total_stock_second_period_123
   , 438.2919691561743 )
 max_lost2_constr_123: Lost_1[123] = MAX ( Lost_1_expr_123 , 0 )
 GC744: Fraction[124] = LOGISTIC ( f_var[124] )
 min_sales_constr_124: Sold_0[124] = MIN ( Q0_var[124] , 1086.899102552989 )
 max_lost_constr_124: Lost_0[124] = MAX ( Lost_0_expr_124 , 0 )
 max_Q1_constr_124: Q1_var[124] = MAX ( Q_hat_adjusted_124 , 0 )
 min_sales2_constr_124: Sold_1[124] = MIN ( total_stock_second_period_124
   , 241.8327568669766 )
 max_lost2_constr_124: Lost_1[124] = MAX ( Lost_1_expr_124 , 0 )
 GC750: Fraction[125] = LOGISTIC ( f_var[125] )
 min_sales_constr_125: Sold_0[125] = MIN ( Q0_var[125] , 1757.520006066862 )
 max_lost_constr_125: Lost_0[125] = MAX ( Lost_0_expr_125 , 0 )
 max_Q1_constr_125: Q1_var[125] = MAX ( Q_hat_adjusted_125 , 0 )
 min_sales2_constr_125: Sold_1[125] = MIN ( total_stock_second_period_125
   , 471.4186035508104 )
 max_lost2_constr_125: Lost_1[125] = MAX ( Lost_1_expr_125 , 0 )
 GC756: Fraction[126] = LOGISTIC ( f_var[126] )
 min_sales_constr_126: Sold_0[126] = MIN ( Q0_var[126] , 1983.283373163777 )
 max_lost_constr_126: Lost_0[126] = MAX ( Lost_0_expr_126 , 0 )
 max_Q1_constr_126: Q1_var[126] = MAX ( Q_hat_adjusted_126 , 0 )
 min_sales2_constr_126: Sold_1[126] = MIN ( total_stock_second_period_126
   , 428.346289448767 )
 max_lost2_constr_126: Lost_1[126] = MAX ( Lost_1_expr_126 , 0 )
 GC762: Fraction[127] = LOGISTIC ( f_var[127] )
 min_sales_constr_127: Sold_0[127] = MIN ( Q0_var[127] , 1573.932562604395 )
 max_lost_constr_127: Lost_0[127] = MAX ( Lost_0_expr_127 , 0 )
 max_Q1_constr_127: Q1_var[127] = MAX ( Q_hat_adjusted_127 , 0 )
 min_sales2_constr_127: Sold_1[127] = MIN ( total_stock_second_period_127
   , 489.2337290844525 )
 max_lost2_constr_127: Lost_1[127] = MAX ( Lost_1_expr_127 , 0 )
 GC768: Fraction[128] = LOGISTIC ( f_var[128] )
 min_sales_constr_128: Sold_0[128] = MIN ( Q0_var[128] , 1719.141299943697 )
 max_lost_constr_128: Lost_0[128] = MAX ( Lost_0_expr_128 , 0 )
 max_Q1_constr_128: Q1_var[128] = MAX ( Q_hat_adjusted_128 , 0 )
 min_sales2_constr_128: Sold_1[128] = MIN ( total_stock_second_period_128
   , 443.2774847387618 )
 max_lost2_constr_128: Lost_1[128] = MAX ( Lost_1_expr_128 , 0 )
 GC774: Fraction[129] = LOGISTIC ( f_var[129] )
 min_sales_constr_129: Sold_0[129] = MIN ( Q0_var[129] , 835.2110498753746 )
 max_lost_constr_129: Lost_0[129] = MAX ( Lost_0_expr_129 , 0 )
 max_Q1_constr_129: Q1_var[129] = MAX ( Q_hat_adjusted_129 , 0 )
 min_sales2_constr_129: Sold_1[129] = MIN ( total_stock_second_period_129
   , 206.3828030264905 )
 max_lost2_constr_129: Lost_1[129] = MAX ( Lost_1_expr_129 , 0 )
 GC780: Fraction[130] = LOGISTIC ( f_var[130] )
 min_sales_constr_130: Sold_0[130] = MIN ( Q0_var[130] , 1093.975868468196 )
 max_lost_constr_130: Lost_0[130] = MAX ( Lost_0_expr_130 , 0 )
 max_Q1_constr_130: Q1_var[130] = MAX ( Q_hat_adjusted_130 , 0 )
 min_sales2_constr_130: Sold_1[130] = MIN ( total_stock_second_period_130
   , 268.4379417137961 )
 max_lost2_constr_130: Lost_1[130] = MAX ( Lost_1_expr_130 , 0 )
 GC786: Fraction[131] = LOGISTIC ( f_var[131] )
 min_sales_constr_131: Sold_0[131] = MIN ( Q0_var[131] , 1209.684516978956 )
 max_lost_constr_131: Lost_0[131] = MAX ( Lost_0_expr_131 , 0 )
 max_Q1_constr_131: Q1_var[131] = MAX ( Q_hat_adjusted_131 , 0 )
 min_sales2_constr_131: Sold_1[131] = MIN ( total_stock_second_period_131
   , 265.1959467506913 )
 max_lost2_constr_131: Lost_1[131] = MAX ( Lost_1_expr_131 , 0 )
 GC792: Fraction[132] = LOGISTIC ( f_var[132] )
 min_sales_constr_132: Sold_0[132] = MIN ( Q0_var[132] , 830.0498175856582 )
 max_lost_constr_132: Lost_0[132] = MAX ( Lost_0_expr_132 , 0 )
 max_Q1_constr_132: Q1_var[132] = MAX ( Q_hat_adjusted_132 , 0 )
 min_sales2_constr_132: Sold_1[132] = MIN ( total_stock_second_period_132
   , 162.7222249858481 )
 max_lost2_constr_132: Lost_1[132] = MAX ( Lost_1_expr_132 , 0 )
 GC798: Fraction[133] = LOGISTIC ( f_var[133] )
 min_sales_constr_133: Sold_0[133] = MIN ( Q0_var[133] , 776.809655078848 )
 max_lost_constr_133: Lost_0[133] = MAX ( Lost_0_expr_133 , 0 )
 max_Q1_constr_133: Q1_var[133] = MAX ( Q_hat_adjusted_133 , 0 )
 min_sales2_constr_133: Sold_1[133] = MIN ( total_stock_second_period_133
   , 166.7634357033703 )
 max_lost2_constr_133: Lost_1[133] = MAX ( Lost_1_expr_133 , 0 )
 GC804: Fraction[134] = LOGISTIC ( f_var[134] )
 min_sales_constr_134: Sold_0[134] = MIN ( Q0_var[134] , 1095.598053054835 )
 max_lost_constr_134: Lost_0[134] = MAX ( Lost_0_expr_134 , 0 )
 max_Q1_constr_134: Q1_var[134] = MAX ( Q_hat_adjusted_134 , 0 )
 min_sales2_constr_134: Sold_1[134] = MIN ( total_stock_second_period_134
   , 287.1590002003493 )
 max_lost2_constr_134: Lost_1[134] = MAX ( Lost_1_expr_134 , 0 )
 GC810: Fraction[135] = LOGISTIC ( f_var[135] )
 min_sales_constr_135: Sold_0[135] = MIN ( Q0_var[135] , 1519.458779768865 )
 max_lost_constr_135: Lost_0[135] = MAX ( Lost_0_expr_135 , 0 )
 max_Q1_constr_135: Q1_var[135] = MAX ( Q_hat_adjusted_135 , 0 )
 min_sales2_constr_135: Sold_1[135] = MIN ( total_stock_second_period_135
   , 384.4696443531508 )
 max_lost2_constr_135: Lost_1[135] = MAX ( Lost_1_expr_135 , 0 )
 GC816: Fraction[136] = LOGISTIC ( f_var[136] )
 min_sales_constr_136: Sold_0[136] = MIN ( Q0_var[136] , 1011.725188404891 )
 max_lost_constr_136: Lost_0[136] = MAX ( Lost_0_expr_136 , 0 )
 max_Q1_constr_136: Q1_var[136] = MAX ( Q_hat_adjusted_136 , 0 )
 min_sales2_constr_136: Sold_1[136] = MIN ( total_stock_second_period_136
   , 201.1041831027184 )
 max_lost2_constr_136: Lost_1[136] = MAX ( Lost_1_expr_136 , 0 )
 GC822: Fraction[137] = LOGISTIC ( f_var[137] )
 min_sales_constr_137: Sold_0[137] = MIN ( Q0_var[137] , 1238.20144609854 )
 max_lost_constr_137: Lost_0[137] = MAX ( Lost_0_expr_137 , 0 )
 max_Q1_constr_137: Q1_var[137] = MAX ( Q_hat_adjusted_137 , 0 )
 min_sales2_constr_137: Sold_1[137] = MIN ( total_stock_second_period_137
   , 295.2257842432172 )
 max_lost2_constr_137: Lost_1[137] = MAX ( Lost_1_expr_137 , 0 )
 GC828: Fraction[138] = LOGISTIC ( f_var[138] )
 min_sales_constr_138: Sold_0[138] = MIN ( Q0_var[138] , 565.4164114734263 )
 max_lost_constr_138: Lost_0[138] = MAX ( Lost_0_expr_138 , 0 )
 max_Q1_constr_138: Q1_var[138] = MAX ( Q_hat_adjusted_138 , 0 )
 min_sales2_constr_138: Sold_1[138] = MIN ( total_stock_second_period_138
   , 147.5452781226486 )
 max_lost2_constr_138: Lost_1[138] = MAX ( Lost_1_expr_138 , 0 )
 GC834: Fraction[139] = LOGISTIC ( f_var[139] )
 min_sales_constr_139: Sold_0[139] = MIN ( Q0_var[139] , 540.9048249119948 )
 max_lost_constr_139: Lost_0[139] = MAX ( Lost_0_expr_139 , 0 )
 max_Q1_constr_139: Q1_var[139] = MAX ( Q_hat_adjusted_139 , 0 )
 min_sales2_constr_139: Sold_1[139] = MIN ( total_stock_second_period_139
   , 109.003643542874 )
 max_lost2_constr_139: Lost_1[139] = MAX ( Lost_1_expr_139 , 0 )
 GC840: Fraction[140] = LOGISTIC ( f_var[140] )
 min_sales_constr_140: Sold_0[140] = MIN ( Q0_var[140] , 720.2539137797929 )
 max_lost_constr_140: Lost_0[140] = MAX ( Lost_0_expr_140 , 0 )
 max_Q1_constr_140: Q1_var[140] = MAX ( Q_hat_adjusted_140 , 0 )
 min_sales2_constr_140: Sold_1[140] = MIN ( total_stock_second_period_140
   , 79.93161392167877 )
 max_lost2_constr_140: Lost_1[140] = MAX ( Lost_1_expr_140 , 0 )
 GC846: Fraction[141] = LOGISTIC ( f_var[141] )
 min_sales_constr_141: Sold_0[141] = MIN ( Q0_var[141] , 2459.008121895637 )
 max_lost_constr_141: Lost_0[141] = MAX ( Lost_0_expr_141 , 0 )
 max_Q1_constr_141: Q1_var[141] = MAX ( Q_hat_adjusted_141 , 0 )
 min_sales2_constr_141: Sold_1[141] = MIN ( total_stock_second_period_141
   , 569.9115957732306 )
 max_lost2_constr_141: Lost_1[141] = MAX ( Lost_1_expr_141 , 0 )
 GC852: Fraction[142] = LOGISTIC ( f_var[142] )
 min_sales_constr_142: Sold_0[142] = MIN ( Q0_var[142] , 356.6318113686676 )
 max_lost_constr_142: Lost_0[142] = MAX ( Lost_0_expr_142 , 0 )
 max_Q1_constr_142: Q1_var[142] = MAX ( Q_hat_adjusted_142 , 0 )
 min_sales2_constr_142: Sold_1[142] = MIN ( total_stock_second_period_142
   , 88.3663385028313 )
 max_lost2_constr_142: Lost_1[142] = MAX ( Lost_1_expr_142 , 0 )
 GC858: Fraction[143] = LOGISTIC ( f_var[143] )
 min_sales_constr_143: Sold_0[143] = MIN ( Q0_var[143] , 826.053118695464 )
 max_lost_constr_143: Lost_0[143] = MAX ( Lost_0_expr_143 , 0 )
 max_Q1_constr_143: Q1_var[143] = MAX ( Q_hat_adjusted_143 , 0 )
 min_sales2_constr_143: Sold_1[143] = MIN ( total_stock_second_period_143
   , 84.4598476537488 )
 max_lost2_constr_143: Lost_1[143] = MAX ( Lost_1_expr_143 , 0 )
 GC864: Fraction[144] = LOGISTIC ( f_var[144] )
 min_sales_constr_144: Sold_0[144] = MIN ( Q0_var[144] , 1359.37020043605 )
 max_lost_constr_144: Lost_0[144] = MAX ( Lost_0_expr_144 , 0 )
 max_Q1_constr_144: Q1_var[144] = MAX ( Q_hat_adjusted_144 , 0 )
 min_sales2_constr_144: Sold_1[144] = MIN ( total_stock_second_period_144
   , 348.6992635764772 )
 max_lost2_constr_144: Lost_1[144] = MAX ( Lost_1_expr_144 , 0 )
 GC870: Fraction[145] = LOGISTIC ( f_var[145] )
 min_sales_constr_145: Sold_0[145] = MIN ( Q0_var[145] , 338.5490396514093 )
 max_lost_constr_145: Lost_0[145] = MAX ( Lost_0_expr_145 , 0 )
 max_Q1_constr_145: Q1_var[145] = MAX ( Q_hat_adjusted_145 , 0 )
 min_sales2_constr_145: Sold_1[145] = MIN ( total_stock_second_period_145
   , 76.53182423382945 )
 max_lost2_constr_145: Lost_1[145] = MAX ( Lost_1_expr_145 , 0 )
 GC876: Fraction[146] = LOGISTIC ( f_var[146] )
 min_sales_constr_146: Sold_0[146] = MIN ( Q0_var[146] , 2137.115918935199 )
 max_lost_constr_146: Lost_0[146] = MAX ( Lost_0_expr_146 , 0 )
 max_Q1_constr_146: Q1_var[146] = MAX ( Q_hat_adjusted_146 , 0 )
 min_sales2_constr_146: Sold_1[146] = MIN ( total_stock_second_period_146
   , 499.7421172731004 )
 max_lost2_constr_146: Lost_1[146] = MAX ( Lost_1_expr_146 , 0 )
 GC882: Fraction[147] = LOGISTIC ( f_var[147] )
 min_sales_constr_147: Sold_0[147] = MIN ( Q0_var[147] , 1875.87637762776 )
 max_lost_constr_147: Lost_0[147] = MAX ( Lost_0_expr_147 , 0 )
 max_Q1_constr_147: Q1_var[147] = MAX ( Q_hat_adjusted_147 , 0 )
 min_sales2_constr_147: Sold_1[147] = MIN ( total_stock_second_period_147
   , 463.7923122432344 )
 max_lost2_constr_147: Lost_1[147] = MAX ( Lost_1_expr_147 , 0 )
 GC888: Fraction[148] = LOGISTIC ( f_var[148] )
 min_sales_constr_148: Sold_0[148] = MIN ( Q0_var[148] , 2696.957938733008 )
 max_lost_constr_148: Lost_0[148] = MAX ( Lost_0_expr_148 , 0 )
 max_Q1_constr_148: Q1_var[148] = MAX ( Q_hat_adjusted_148 , 0 )
 min_sales2_constr_148: Sold_1[148] = MIN ( total_stock_second_period_148
   , 592.1020474743182 )
 max_lost2_constr_148: Lost_1[148] = MAX ( Lost_1_expr_148 , 0 )
 GC894: Fraction[149] = LOGISTIC ( f_var[149] )
 min_sales_constr_149: Sold_0[149] = MIN ( Q0_var[149] , 1641.775307931028 )
 max_lost_constr_149: Lost_0[149] = MAX ( Lost_0_expr_149 , 0 )
 max_Q1_constr_149: Q1_var[149] = MAX ( Q_hat_adjusted_149 , 0 )
 min_sales2_constr_149: Sold_1[149] = MIN ( total_stock_second_period_149
   , 346.671016981829 )
 max_lost2_constr_149: Lost_1[149] = MAX ( Lost_1_expr_149 , 0 )
End
