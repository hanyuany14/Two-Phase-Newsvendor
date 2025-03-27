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
   + profits_vars[149] + profits_vars[150] + profits_vars[151]
   + profits_vars[152] + profits_vars[153] + profits_vars[154]
   + profits_vars[155] + profits_vars[156] + profits_vars[157]
   + profits_vars[158] + profits_vars[159] + profits_vars[160]
   + profits_vars[161] + profits_vars[162] + profits_vars[163]
   + profits_vars[164] + profits_vars[165] + profits_vars[166]
   + profits_vars[167] + profits_vars[168] + profits_vars[169]
   + profits_vars[170] + profits_vars[171] + profits_vars[172]
   + profits_vars[173] + profits_vars[174] + profits_vars[175]
   + profits_vars[176] + profits_vars[177] + profits_vars[178]
   + profits_vars[179] + profits_vars[180] + profits_vars[181]
   + profits_vars[182] + profits_vars[183] + profits_vars[184]
   + profits_vars[185] + profits_vars[186] + profits_vars[187]
   + profits_vars[188] + profits_vars[189] + profits_vars[190]
   + profits_vars[191] + profits_vars[192] + profits_vars[193]
   + profits_vars[194] + profits_vars[195] + profits_vars[196]
   + profits_vars[197] + profits_vars[198] + profits_vars[199]
   + profits_vars[200] + profits_vars[201] + profits_vars[202]
   + profits_vars[203] + profits_vars[204] + profits_vars[205]
   + profits_vars[206] + profits_vars[207] + profits_vars[208]
   + profits_vars[209] + profits_vars[210] + profits_vars[211]
   + profits_vars[212] + profits_vars[213] + profits_vars[214]
   + profits_vars[215] + profits_vars[216] + profits_vars[217]
   + profits_vars[218] + profits_vars[219] + profits_vars[220]
   + profits_vars[221] + profits_vars[222] + profits_vars[223]
   + profits_vars[224] + profits_vars[225] + profits_vars[226]
   + profits_vars[227] + profits_vars[228] + profits_vars[229]
   + profits_vars[230] + profits_vars[231] + profits_vars[232]
   + profits_vars[233] + profits_vars[234] + profits_vars[235]
   + profits_vars[236] + profits_vars[237] + profits_vars[238]
   + profits_vars[239] + profits_vars[240] + profits_vars[241]
   + profits_vars[242] + profits_vars[243] + profits_vars[244]
   + profits_vars[245] + profits_vars[246] + profits_vars[247]
   + profits_vars[248] + profits_vars[249]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: 0.5959541390893992 alphas[0] + 0.6367483818195703 alphas[1]
   - 0.8957475709089285 alphas[2] + 1.118378197902526 alphas[3] - alphas[4]
   + f_var[0] = 0
 R3: - 2094.959652373807 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 2209.591685417078
 R5: Q0_var[0] + Q_hat_adjusted_0 = 2677.0704226487
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 467.4681275293227
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: 0.3824338004940984 alphas[0] - 0.6095234655271226 alphas[1]
   - 0.2444290075395085 alphas[2] + 1.430401821645257 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2094.959652373807 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 1460.924674130369
 R15: Q0_var[1] + Q_hat_adjusted_1 = 1835.788067484745
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 374.8527808345601
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.1353514409021558 alphas[0] - 1.294920067840932 alphas[1]
   - 0.4125532792257035 alphas[2] + 0.130960814103319 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 2094.959652373807 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1956.876720188578
 R25: Q0_var[2] + Q_hat_adjusted_2 = 2355.646828431188
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 398.759496512169
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.9042633654178789 alphas[0] + 1.311011427507537 alphas[1]
   + 1.558299916499328 alphas[2] - 0.1822655609174497 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 2094.959652373807 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 574.4938879510611
 R35: Q0_var[3] + Q_hat_adjusted_3 = 693.0151910742649
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 118.5106869860278
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.5171402568792384 alphas[0] - 0.3529661763953919 alphas[1]
   + 1.57786997223634 alphas[2] - 1.17495388876262 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2094.959652373807 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 350.5583856119644
 R45: Q0_var[4] + Q_hat_adjusted_4 = 466.2968907798883
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 115.7278896878987
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 1.495755798507852 alphas[0] - 0.5802577965516729 alphas[1]
   - 1.167182674135675 alphas[2] - 0.3028384368082361 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2094.959652373807 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1930.424972286104
 R55: Q0_var[5] + Q_hat_adjusted_5 = 2436.50088760744
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 506.0653019449017
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: - 1.285008892009202 alphas[0] + 0.7539420242461605 alphas[1]
   + 0.6650478105794233 alphas[2] - 0.3380916920300534 alphas[3]
   - alphas[4] + f_var[6] = 0
 R63: - 2094.959652373807 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 1301.7713384587
 R65: Q0_var[6] + Q_hat_adjusted_6 = 1547.310135052977
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 245.5281836893697
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: 1.487776157999749 alphas[0] - 0.8408617423399116 alphas[1]
   - 1.149895544702414 alphas[2] + 0.6121903160498761 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2094.959652373807 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 2154.361153193902
 R75: Q0_var[7] + Q_hat_adjusted_7 = 2657.978892369054
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 503.6071292842974
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: 0.7893140852491326 alphas[0] + 1.327679039255043 alphas[1]
   - 0.7852241439487312 alphas[2] + 0.6840549937035069 alphas[3]
   - alphas[4] + f_var[8] = 0
 R83: - 2094.959652373807 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1788.712843767133
 R85: Q0_var[8] + Q_hat_adjusted_8 = 2240.475517092824
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 451.7520615648307
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.1333121827718576 alphas[0] + 0.8750969406393045 alphas[1]
   - 0.2980241459964683 alphas[2] + 0.6792459336676766 alphas[3]
   - alphas[4] + f_var[9] = 0
 R93: - 2094.959652373807 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 1470.725776623855
 R95: Q0_var[9] + Q_hat_adjusted_9 = 1853.210222243987
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 382.4738323575915
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: 1.297144968886463 alphas[0] + 1.669027263431656 alphas[1]
   + 1.319407019576068 alphas[2] + 1.569741421970016 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 2094.959652373807 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 559.9662753110898
 R105: Q0_var[10] + Q_hat_adjusted_10 = 712.457357406801
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 152.4804672376567
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 0.4661286340369206 alphas[0] + 0.0019179350567592 alphas[1]
   + 1.456254541120253 alphas[2] + 0.7352051056226956 alphas[3] - alphas[4]
   + f_var[11] = 0
 R113: - 2094.959652373807 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 570.8335332569737
 R115: Q0_var[11] + Q_hat_adjusted_11 = 703.8653506797555
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 133.0212020421171
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: - 0.8684886690789593 alphas[0] + 0.7962334209055841 alphas[1]
   + 0.3382992803657388 alphas[2] - 1.065799443962019 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 2094.959652373807 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 1220.288998083456
 R125: Q0_var[12] + Q_hat_adjusted_12 = 1512.290355041136
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 291.9907445769216
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 0.6654340495039907 alphas[0] + 1.41152740097662 alphas[1]
   - 1.564946109429916 alphas[2] - 1.069874369459572 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 2094.959652373807 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 2208.330690747674
 R135: Q0_var[13] + Q_hat_adjusted_13 = 2770.967248037876
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 562.6259475861058
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: - 0.4673311191168009 alphas[0] - 1.253254672482076 alphas[1]
   + 0.8928767785570539 alphas[2] - 0.8971673930340176 alphas[3]
   - alphas[4] + f_var[14] = 0
 R143: - 2094.959652373807 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 926.6836858879103
 R145: Q0_var[14] + Q_hat_adjusted_14 = 1139.825959709978
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 213.1316575590632
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: - 0.7677192798408541 alphas[0] + 1.011227834015958 alphas[1]
   + 0.2037699267324032 alphas[2] - 0.5361020196403654 alphas[3]
   - alphas[4] + f_var[15] = 0
 R153: - 2094.959652373807 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 1333.071048766716
 R155: Q0_var[15] + Q_hat_adjusted_15 = 1644.202036165387
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 311.1203739596893
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 0.9553442883159851 alphas[0] - 0.2124488027335724 alphas[1]
   - 0.56786210318319 alphas[2] - 1.370602957038503 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 2094.959652373807 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1458.825881873058
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1879.68039202153
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 420.8438978858483
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: 0.4892271291372569 alphas[0] + 0.7695811680314901 alphas[1]
   - 0.0580544134441237 alphas[2] - 1.325679133170132 alphas[3] - alphas[4]
   + f_var[17] = 0
 R173: - 2094.959652373807 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1518.131641077346
 R175: Q0_var[17] + Q_hat_adjusted_17 = 1866.493183596534
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 348.3509296391269
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.6508753925559214 alphas[0] - 0.3126246311327072 alphas[1]
   + 1.600232747680812 alphas[2] + 0.3837972061532688 alphas[3] - alphas[4]
   + f_var[18] = 0
 R183: - 2094.959652373807 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 409.344249626055
 R185: Q0_var[18] + Q_hat_adjusted_18 = 521.9028413863595
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 112.5479770271992
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.342658108736612 alphas[0] - 0.5011512209139125 alphas[1]
   - 0.7563288929050397 alphas[2] - 0.6173431015715913 alphas[3]
   - alphas[4] + f_var[19] = 0
 R193: - 2094.959652373807 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 2079.269866385042
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2526.923730077167
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 447.6432523833461
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 0.1991056009112765 alphas[0] - 0.7377997239927726 alphas[1]
   - 1.540880076496122 alphas[2] - 1.396011140871272 alphas[3] - alphas[4]
   + f_var[20] = 0
 R203: - 2094.959652373807 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 2051.126239662301
 R205: Q0_var[20] + Q_hat_adjusted_20 = 2610.340685443143
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 559.2038372562537
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: 0.6902895413084772 alphas[0] + 0.6890527485976006 alphas[1]
   - 0.4202022093525224 alphas[2] - 0.8812330050551653 alphas[3]
   - alphas[4] + f_var[21] = 0
 R213: - 2094.959652373807 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1341.583632819008
 R215: Q0_var[21] + Q_hat_adjusted_21 = 1741.441393376518
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 399.8471490916911
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 1.550669565800169 alphas[0] + 1.513802910357796 alphas[1]
   - 0.3524031121097896 alphas[2] - 1.536937419947184 alphas[3] - alphas[4]
   + f_var[22] = 0
 R223: - 2094.959652373807 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 1378.224472273111
 R225: Q0_var[22] + Q_hat_adjusted_22 = 1768.4414256462
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 390.2063415823155
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 0.7670664106413965 alphas[0] - 1.174886596826701 alphas[1]
   - 1.248646022695326 alphas[2] - 1.721462499140321 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2094.959652373807 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 2391.879272655814
 R235: Q0_var[23] + Q_hat_adjusted_23 = 2909.539003119016
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 517.649120822622
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 1.626840651275248 alphas[0] - 1.546721909447219 alphas[1]
   - 1.239996602175277 alphas[2] - 0.060365804703595 alphas[3] - alphas[4]
   + f_var[24] = 0
 R243: - 2094.959652373807 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 1971.702694486961
 R245: Q0_var[24] + Q_hat_adjusted_24 = 2488.132508492506
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 516.4192018164308
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: 0.3836463203994954 alphas[0] - 1.270990583491689 alphas[1]
   + 1.031792762245722 alphas[2] - 0.374236383067958 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 2094.959652373807 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 618.6502701757317
 R255: Q0_var[25] + Q_hat_adjusted_25 = 812.0391477045837
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 193.3782638824937
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: - 0.1543225902170167 alphas[0] + 1.396449218715601 alphas[1]
   + 0.2035645800898881 alphas[2] + 0.1161091408598054 alphas[3]
   - alphas[4] + f_var[26] = 0
 R263: - 2094.959652373807 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 1381.884732282544
 R265: Q0_var[26] + Q_hat_adjusted_26 = 1693.044919008105
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 311.1495735734555
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.488778734615603 alphas[0] - 0.9354472411645491 alphas[1]
   - 1.192285853075731 alphas[2] + 0.1956425263265538 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 2094.959652373807 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 2235.997420321601
 R275: Q0_var[27] + Q_hat_adjusted_27 = 2745.642924869043
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 509.6348910084261
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: 1.348718287721178 alphas[0] + 0.057464636123952 alphas[1]
   - 0.8953878994001155 alphas[2] + 1.418747821729026 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 2094.959652373807 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 1812.715536432553
 R285: Q0_var[28] + Q_hat_adjusted_28 = 2280.143131633212
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 467.4169834297094
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.2702876112210127 alphas[0] - 0.4275447133345647 alphas[1]
   - 1.280608062899451 alphas[2] + 0.4368291824697957 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 2094.959652373807 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 2084.481256144275
 R295: Q0_var[29] + Q_hat_adjusted_29 = 2606.685885943466
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 522.1940172850586
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 1.570851503547021 alphas[0] - 0.3598107498992543 alphas[1]
   + 1.233320720063562 alphas[2] + 0.1537610224386337 alphas[3] - alphas[4]
   + f_var[30] = 0
 R303: - 2094.959652373807 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 653.2976280459978
 R305: Q0_var[30] + Q_hat_adjusted_30 = 818.0298952589679
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 164.7216544587017
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 700 Sold_0[30] - 700 Sold_1[30] + 700 Lost_0[30] + 700 Lost_1[30]
   + 300 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: 0.4861161373109948 alphas[0] - 0.1052735242970388 alphas[1]
   + 0.1431977975196898 alphas[2] - 0.3742311160629526 alphas[3]
   - alphas[4] + f_var[31] = 0
 R313: - 2094.959652373807 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 1033.924702446921
 R315: Q0_var[31] + Q_hat_adjusted_31 = 1353.668845109145
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 319.7335305048583
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 700 Sold_0[31] - 700 Sold_1[31] + 700 Lost_0[31] + 700 Lost_1[31]
   + 300 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: - 0.4668084464053881 alphas[0] - 0.5701306514171819 alphas[1]
   - 0.8043845553565028 alphas[2] + 1.290006712428225 alphas[3] - alphas[4]
   + f_var[32] = 0
 R323: - 2094.959652373807 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 1591.035565029601
 R325: Q0_var[32] + Q_hat_adjusted_32 = 2045.522785672114
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 454.4766087298385
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 700 Sold_0[32] - 700 Sold_1[32] + 700 Lost_0[32] + 700 Lost_1[32]
   + 300 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: 0.7026661162233164 alphas[0] - 0.3751000454652365 alphas[1]
   - 0.8789574455548006 alphas[2] - 1.735803635737621 alphas[3] - alphas[4]
   + f_var[33] = 0
 R333: - 2094.959652373807 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 1894.230392865846
 R335: Q0_var[33] + Q_hat_adjusted_33 = 2359.321630504899
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 465.080627205861
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 700 Sold_0[33] - 700 Sold_1[33] + 700 Lost_0[33] + 700 Lost_1[33]
   + 300 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: 0.9388967411265293 alphas[0] - 0.2440518609350668 alphas[1]
   + 1.4359920678053 alphas[2] + 0.4572824722322433 alphas[3] - alphas[4]
   + f_var[34] = 0
 R343: - 2094.959652373807 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 680.50994738089
 R345: Q0_var[34] + Q_hat_adjusted_34 = 816.4230195704654
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 135.9024587626776
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 700 Sold_0[34] - 700 Sold_1[34] + 700 Lost_0[34] + 700 Lost_1[34]
   + 300 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: 1.259401011673255 alphas[0] + 1.19657915845752 alphas[1]
   - 0.8148881013894168 alphas[2] + 0.832707688073909 alphas[3] - alphas[4]
   + f_var[35] = 0
 R353: - 2094.959652373807 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 1910.45053084193
 R355: Q0_var[35] + Q_hat_adjusted_35 = 2366.431319922307
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 455.9701782549112
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 700 Sold_0[35] - 700 Sold_1[35] + 700 Lost_0[35] + 700 Lost_1[35]
   + 300 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 0.3131422762694672 alphas[0] - 1.20033170476964 alphas[1]
   - 0.5135031160224947 alphas[2] + 0.2864944536731928 alphas[3]
   - alphas[4] + f_var[36] = 0
 R363: - 2094.959652373807 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 1644.206380798092
 R365: Q0_var[36] + Q_hat_adjusted_36 = 2057.331221320957
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 413.1142296021448
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 700 Sold_0[36] - 700 Sold_1[36] + 700 Lost_0[36] + 700 Lost_1[36]
   + 300 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.7157197892401148 alphas[0] - 0.8304219616946525 alphas[1]
   + 0.4616087939110346 alphas[2] + 0.0392565757111758 alphas[3]
   - alphas[4] + f_var[37] = 0
 R373: - 2094.959652373807 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 1114.189229344849
 R375: Q0_var[37] + Q_hat_adjusted_37 = 1388.656381464894
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 274.4565393660645
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 700 Sold_0[37] - 700 Sold_1[37] + 700 Lost_0[37] + 700 Lost_1[37]
   + 300 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: - 0.6362712642744018 alphas[0] - 1.530375718052384 alphas[1]
   + 0.6191528203599299 alphas[2] + 0.5232786883804639 alphas[3]
   - alphas[4] + f_var[38] = 0
 R383: - 2094.959652373807 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 1273.167012356758
 R385: Q0_var[38] + Q_hat_adjusted_38 = 1525.231924089865
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 252.0542995868327
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 700 Sold_0[38] - 700 Sold_1[38] + 700 Lost_0[38] + 700 Lost_1[38]
   + 300 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 1.579939257964751 alphas[0] - 0.9573772343343013 alphas[1]
   + 1.586472246075562 alphas[2] - 0.9844366092380198 alphas[3] - alphas[4]
   + f_var[39] = 0
 R393: - 2094.959652373807 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 637.2819340078738
 R395: Q0_var[39] + Q_hat_adjusted_39 = 751.7972254003537
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 114.5046741567195
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 700 Sold_0[39] - 700 Sold_1[39] + 700 Lost_0[39] + 700 Lost_1[39]
   + 300 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: 1.644496740840545 alphas[0] - 0.2629071140696234 alphas[1]
   + 0.3101399566897214 alphas[2] - 1.636115839717457 alphas[3] - alphas[4]
   + f_var[40] = 0
 R403: - 2094.959652373807 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 1102.71055919792
 R405: Q0_var[40] + Q_hat_adjusted_40 = 1398.716079508157
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 295.9949073194246
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 700 Sold_0[40] - 700 Sold_1[40] + 700 Lost_0[40] + 700 Lost_1[40]
   + 300 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: - 0.5518893524061027 alphas[0] - 1.373699189444693 alphas[1]
   + 0.6090031172415615 alphas[2] + 0.9478568513773489 alphas[3]
   - alphas[4] + f_var[41] = 0
 R413: - 2094.959652373807 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 1009.456615620397
 R415: Q0_var[41] + Q_hat_adjusted_41 = 1262.964782968015
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 253.4975538191565
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 700 Sold_0[41] - 700 Sold_1[41] + 700 Lost_0[41] + 700 Lost_1[41]
   + 300 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.3923833531611147 alphas[0] - 0.5010386133706688 alphas[1]
   + 0.1712072033525724 alphas[2] + 1.576004042238726 alphas[3] - alphas[4]
   + f_var[42] = 0
 R423: - 2094.959652373807 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 1041.928554297051
 R425: Q0_var[42] + Q_hat_adjusted_42 = 1357.68985279797
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 315.7506855840766
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 700 Sold_0[42] - 700 Sold_1[42] + 700 Lost_0[42] + 700 Lost_1[42]
   + 300 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: - 0.3526720596337793 alphas[0] + 0.6412428133394186 alphas[1]
   - 0.8036545770050523 alphas[2] - 0.2661137666320889 alphas[3]
   - alphas[4] + f_var[43] = 0
 R433: - 2094.959652373807 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 1505.032168234075
 R435: Q0_var[43] + Q_hat_adjusted_43 = 1959.41558597758
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 454.3728082212985
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 700 Sold_0[43] - 700 Sold_1[43] + 700 Lost_0[43] + 700 Lost_1[43]
   + 300 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: - 1.627341123146449 alphas[0] + 1.08810808275255 alphas[1]
   + 1.176924897687498 alphas[2] - 0.5271116517135164 alphas[3] - alphas[4]
   + f_var[44] = 0
 R443: - 2094.959652373807 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 625.9527028651543
 R445: Q0_var[44] + Q_hat_adjusted_44 = 798.7042746652382
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 172.740953977432
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 700 Sold_0[44] - 700 Sold_1[44] + 700 Lost_0[44] + 700 Lost_1[44]
   + 300 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: 1.596211676283536 alphas[0] - 0.3352231150074202 alphas[1]
   + 0.000805701 alphas[2] - 1.291309239017716 alphas[3] - alphas[4]
   + f_var[45] = 0
 R453: - 2094.959652373807 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 1386.150798236841
 R455: Q0_var[45] + Q_hat_adjusted_45 = 1726.142626230548
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 339.9812159257486
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 700 Sold_0[45] - 700 Sold_1[45] + 700 Lost_0[45] + 700 Lost_1[45]
   + 300 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: - 0.7198516434956336 alphas[0] - 0.7729273766203792 alphas[1]
   - 1.099398694727458 alphas[2] - 1.464258684141576 alphas[3] - alphas[4]
   + f_var[46] = 0
 R463: - 2094.959652373807 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 1464.285830788816
 R465: Q0_var[46] + Q_hat_adjusted_46 = 1960.723084360243
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 496.4266441416203
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 700 Sold_0[46] - 700 Sold_1[46] + 700 Lost_0[46] + 700 Lost_1[46]
   + 300 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: - 1.119247711193984 alphas[0] - 1.064986123156 alphas[1]
   + 0.814213643302797 alphas[2] + 1.697082385899279 alphas[3] - alphas[4]
   + f_var[47] = 0
 R473: - 2094.959652373807 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 1017.000072266852
 R475: Q0_var[47] + Q_hat_adjusted_47 = 1241.327980636207
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 224.3172953627362
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 700 Sold_0[47] - 700 Sold_1[47] + 700 Lost_0[47] + 700 Lost_1[47]
   + 300 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: - 0.0836762185196377 alphas[0] + 0.7237000853124373 alphas[1]
   - 1.348458975489389 alphas[2] + 1.100315505611073 alphas[3] - alphas[4]
   + f_var[48] = 0
 R483: - 2094.959652373807 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 2138.230680416611
 R485: Q0_var[48] + Q_hat_adjusted_48 = 2670.083482203914
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 531.8421927655245
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 700 Sold_0[48] - 700 Sold_1[48] + 700 Lost_0[48] + 700 Lost_1[48]
   + 300 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: 0.5155814584569119 alphas[0] - 1.323353976492631 alphas[1]
   - 1.592047895627965 alphas[2] + 0.582518608573654 alphas[3] - alphas[4]
   + f_var[49] = 0
 R493: - 2094.959652373807 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 2230.372189250569
 R495: Q0_var[49] + Q_hat_adjusted_49 = 2796.862529387687
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 566.4797319950662
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 700 Sold_0[49] - 700 Sold_1[49] + 700 Lost_0[49] + 700 Lost_1[49]
   + 300 Left_1[49] + profits_vars[49] = 0
 Fraction_lower_bound_50: Fraction[50] >= 0
 Fraction_upper_bound_50: Fraction[50] <= 1
 R502: - 1.041415834296586 alphas[0] - 0.2478043072882619 alphas[1]
   + 1.291703581591962 alphas[2] + 0.6345669484101907 alphas[3] - alphas[4]
   + f_var[50] = 0
 R503: - 2094.959652373807 Fraction[50] + Q0_var[50] = 0
 R504: Q0_var[50] + Lost_0_expr_50 = 721.1527100297592
 R505: Q0_var[50] + Q_hat_adjusted_50 = 877.5831289025557
 R506: Sold_0[50] - Q0_var[50] - Q1_var[50] + total_stock_second_period_50
   = 0
 R507: total_stock_second_period_50 + Lost_1_expr_50 = 156.4198045414643
 R508: Sold_1[50] + Left_1[50] - total_stock_second_period_50 = 0
 R509: - 700 Sold_0[50] - 700 Sold_1[50] + 700 Lost_0[50] + 700 Lost_1[50]
   + 300 Left_1[50] + profits_vars[50] = 0
 Fraction_lower_bound_51: Fraction[51] >= 0
 Fraction_upper_bound_51: Fraction[51] <= 1
 R512: 1.322312378308099 alphas[0] - 0.7295073203721373 alphas[1]
   + 1.281643975940591 alphas[2] - 0.00845035678825892 alphas[3]
   - alphas[4] + f_var[51] = 0
 R513: - 2094.959652373807 Fraction[51] + Q0_var[51] = 0
 R514: Q0_var[51] + Lost_0_expr_51 = 514.7704737637375
 R515: Q0_var[51] + Q_hat_adjusted_51 = 672.6313378475148
 R516: Sold_0[51] - Q0_var[51] - Q1_var[51] + total_stock_second_period_51
   = 0
 R517: total_stock_second_period_51 + Lost_1_expr_51 = 157.8502472190389
 R518: Sold_1[51] + Left_1[51] - total_stock_second_period_51 = 0
 R519: - 700 Sold_0[51] - 700 Sold_1[51] + 700 Lost_0[51] + 700 Lost_1[51]
   + 300 Left_1[51] + profits_vars[51] = 0
 Fraction_lower_bound_52: Fraction[52] >= 0
 Fraction_upper_bound_52: Fraction[52] <= 1
 R522: 0.1472842540449275 alphas[0] - 1.011780029220986 alphas[1]
   - 0.5802189323004743 alphas[2] + 1.241199223735836 alphas[3] - alphas[4]
   + f_var[52] = 0
 R523: - 2094.959652373807 Fraction[52] + Q0_var[52] = 0
 R524: Q0_var[52] + Lost_0_expr_52 = 1747.116130576178
 R525: Q0_var[52] + Q_hat_adjusted_52 = 2169.727740574822
 R526: Sold_0[52] - Q0_var[52] - Q1_var[52] + total_stock_second_period_52
   = 0
 R527: total_stock_second_period_52 + Lost_1_expr_52 = 422.6009981392555
 R528: Sold_1[52] + Left_1[52] - total_stock_second_period_52 = 0
 R529: - 700 Sold_0[52] - 700 Sold_1[52] + 700 Lost_0[52] + 700 Lost_1[52]
   + 300 Left_1[52] + profits_vars[52] = 0
 Fraction_lower_bound_53: Fraction[53] >= 0
 Fraction_upper_bound_53: Fraction[53] <= 1
 R532: - 0.1182502364665638 alphas[0] - 1.420549255593757 alphas[1]
   - 0.1300959242586968 alphas[2] - 1.259935289988275 alphas[3] - alphas[4]
   + f_var[53] = 0
 R533: - 2094.959652373807 Fraction[53] + Q0_var[53] = 0
 R534: Q0_var[53] + Lost_0_expr_53 = 1395.194890639617
 R535: Q0_var[53] + Q_hat_adjusted_53 = 1753.800497551149
 R536: Sold_0[53] - Q0_var[53] - Q1_var[53] + total_stock_second_period_53
   = 0
 R537: total_stock_second_period_53 + Lost_1_expr_53 = 358.5949943345804
 R538: Sold_1[53] + Left_1[53] - total_stock_second_period_53 = 0
 R539: - 700 Sold_0[53] - 700 Sold_1[53] + 700 Lost_0[53] + 700 Lost_1[53]
   + 300 Left_1[53] + profits_vars[53] = 0
 Fraction_lower_bound_54: Fraction[54] >= 0
 Fraction_upper_bound_54: Fraction[54] <= 1
 R542: - 0.7112431544980005 alphas[0] + 0.7080913740736664 alphas[1]
   - 1.488733644584705 alphas[2] - 0.1060876238810721 alphas[3] - alphas[4]
   + f_var[54] = 0
 R543: - 2094.959652373807 Fraction[54] + Q0_var[54] = 0
 R544: Q0_var[54] + Lost_0_expr_54 = 2121.712027700773
 R545: Q0_var[54] + Q_hat_adjusted_54 = 2673.511424671411
 R546: Sold_0[54] - Q0_var[54] - Q1_var[54] + total_stock_second_period_54
   = 0
 R547: total_stock_second_period_54 + Lost_1_expr_54 = 551.7887870151146
 R548: Sold_1[54] + Left_1[54] - total_stock_second_period_54 = 0
 R549: - 700 Sold_0[54] - 700 Sold_1[54] + 700 Lost_0[54] + 700 Lost_1[54]
   + 300 Left_1[54] + profits_vars[54] = 0
 Fraction_lower_bound_55: Fraction[55] >= 0
 Fraction_upper_bound_55: Fraction[55] <= 1
 R552: 1.343977225067654 alphas[0] + 0.3679416784388554 alphas[1]
   - 1.131391762550023 alphas[2] - 1.737835895723848 alphas[3] - alphas[4]
   + f_var[55] = 0
 R553: - 2094.959652373807 Fraction[55] + Q0_var[55] = 0
 R554: Q0_var[55] + Lost_0_expr_55 = 1503.617297778341
 R555: Q0_var[55] + Q_hat_adjusted_55 = 2004.603861839825
 R556: Sold_0[55] - Q0_var[55] - Q1_var[55] + total_stock_second_period_55
   = 0
 R557: total_stock_second_period_55 + Lost_1_expr_55 = 500.9759526288333
 R558: Sold_1[55] + Left_1[55] - total_stock_second_period_55 = 0
 R559: - 700 Sold_0[55] - 700 Sold_1[55] + 700 Lost_0[55] + 700 Lost_1[55]
   + 300 Left_1[55] + profits_vars[55] = 0
 Fraction_lower_bound_56: Fraction[56] >= 0
 Fraction_upper_bound_56: Fraction[56] <= 1
 R562: 0.8775215170440444 alphas[0] - 1.316606379417568 alphas[1]
   - 0.0763062531541912 alphas[2] - 0.4651841109814174 alphas[3]
   - alphas[4] + f_var[56] = 0
 R563: - 2094.959652373807 Fraction[56] + Q0_var[56] = 0
 R564: Q0_var[56] + Lost_0_expr_56 = 1528.813481598439
 R565: Q0_var[56] + Q_hat_adjusted_56 = 1879.770374165903
 R566: Sold_0[56] - Q0_var[56] - Q1_var[56] + total_stock_second_period_56
   = 0
 R567: total_stock_second_period_56 + Lost_1_expr_56 = 350.9462809121204
 R568: Sold_1[56] + Left_1[56] - total_stock_second_period_56 = 0
 R569: - 700 Sold_0[56] - 700 Sold_1[56] + 700 Lost_0[56] + 700 Lost_1[56]
   + 300 Left_1[56] + profits_vars[56] = 0
 Fraction_lower_bound_57: Fraction[57] >= 0
 Fraction_upper_bound_57: Fraction[57] <= 1
 R572: 0.2554316404627066 alphas[0] - 1.099221415995163 alphas[1]
   + 1.446943126371665 alphas[2] + 0.2941279664328442 alphas[3] - alphas[4]
   + f_var[57] = 0
 R573: - 2094.959652373807 Fraction[57] + Q0_var[57] = 0
 R574: Q0_var[57] + Lost_0_expr_57 = 558.006290834075
 R575: Q0_var[57] + Q_hat_adjusted_57 = 692.3621592231414
 R576: Sold_0[57] - Q0_var[57] - Q1_var[57] + total_stock_second_period_57
   = 0
 R577: total_stock_second_period_57 + Lost_1_expr_57 = 134.3452543819624
 R578: Sold_1[57] + Left_1[57] - total_stock_second_period_57 = 0
 R579: - 700 Sold_0[57] - 700 Sold_1[57] + 700 Lost_0[57] + 700 Lost_1[57]
   + 300 Left_1[57] + profits_vars[57] = 0
 Fraction_lower_bound_58: Fraction[58] >= 0
 Fraction_upper_bound_58: Fraction[58] <= 1
 R582: - 0.4704331650592617 alphas[0] + 0.3458031774023599 alphas[1]
   + 1.340822623161423 alphas[2] + 0.5636009470191928 alphas[3] - alphas[4]
   + f_var[58] = 0
 R583: - 2094.959652373807 Fraction[58] + Q0_var[58] = 0
 R584: Q0_var[58] + Lost_0_expr_58 = 862.2784349300385
 R585: Q0_var[58] + Q_hat_adjusted_58 = 1011.72428688563
 R586: Sold_0[58] - Q0_var[58] - Q1_var[58] + total_stock_second_period_58
   = 0
 R587: total_stock_second_period_58 + Lost_1_expr_58 = 149.4352391997712
 R588: Sold_1[58] + Left_1[58] - total_stock_second_period_58 = 0
 R589: - 700 Sold_0[58] - 700 Sold_1[58] + 700 Lost_0[58] + 700 Lost_1[58]
   + 300 Left_1[58] + profits_vars[58] = 0
 Fraction_lower_bound_59: Fraction[59] >= 0
 Fraction_upper_bound_59: Fraction[59] <= 1
 R592: - 1.368987808694591 alphas[0] + 0.1194200681719037 alphas[1]
   + 0.1114672784899743 alphas[2] - 1.325498055659327 alphas[3] - alphas[4]
   + f_var[59] = 0
 R593: - 2094.959652373807 Fraction[59] + Q0_var[59] = 0
 R594: Q0_var[59] + Lost_0_expr_59 = 1582.803976715511
 R595: Q0_var[59] + Q_hat_adjusted_59 = 1907.060094585178
 R596: Sold_0[59] - Q0_var[59] - Q1_var[59] + total_stock_second_period_59
   = 0
 R597: total_stock_second_period_59 + Lost_1_expr_59 = 324.2455054216166
 R598: Sold_1[59] + Left_1[59] - total_stock_second_period_59 = 0
 R599: - 700 Sold_0[59] - 700 Sold_1[59] + 700 Lost_0[59] + 700 Lost_1[59]
   + 300 Left_1[59] + profits_vars[59] = 0
 Fraction_lower_bound_60: Fraction[60] >= 0
 Fraction_upper_bound_60: Fraction[60] <= 1
 R602: - 0.4552048304564242 alphas[0] - 0.7097726472081827 alphas[1]
   + 0.6175247240750951 alphas[2] - 1.37072249438539 alphas[3] - alphas[4]
   + f_var[60] = 0
 R603: - 2094.959652373807 Fraction[60] + Q0_var[60] = 0
 R604: Q0_var[60] + Lost_0_expr_60 = 810.8758598655588
 R605: Q0_var[60] + Q_hat_adjusted_60 = 1063.172282919032
 R606: Sold_0[60] - Q0_var[60] - Q1_var[60] + total_stock_second_period_60
   = 0
 R607: total_stock_second_period_60 + Lost_1_expr_60 = 252.285809496679
 R608: Sold_1[60] + Left_1[60] - total_stock_second_period_60 = 0
 R609: - 700 Sold_0[60] - 700 Sold_1[60] + 700 Lost_0[60] + 700 Lost_1[60]
   + 300 Left_1[60] + profits_vars[60] = 0
 Fraction_lower_bound_61: Fraction[61] >= 0
 Fraction_upper_bound_61: Fraction[61] <= 1
 R612: 1.198882248096082 alphas[0] + 1.467350846203866 alphas[1]
   - 1.627808099576866 alphas[2] + 0.8748230290818152 alphas[3] - alphas[4]
   + f_var[61] = 0
 R613: - 2094.959652373807 Fraction[61] + Q0_var[61] = 0
 R614: Q0_var[61] + Lost_0_expr_61 = 2163.019327614861
 R615: Q0_var[61] + Q_hat_adjusted_61 = 2734.594652170801
 R616: Sold_0[61] - Q0_var[61] - Q1_var[61] + total_stock_second_period_61
   = 0
 R617: total_stock_second_period_61 + Lost_1_expr_61 = 571.5647147866965
 R618: Sold_1[61] + Left_1[61] - total_stock_second_period_61 = 0
 R619: - 700 Sold_0[61] - 700 Sold_1[61] + 700 Lost_0[61] + 700 Lost_1[61]
   + 300 Left_1[61] + profits_vars[61] = 0
 Fraction_lower_bound_62: Fraction[62] >= 0
 Fraction_upper_bound_62: Fraction[62] <= 1
 R622: - 1.68803565167442 alphas[0] + 0.4514154959021287 alphas[1]
   - 0.65280791224865 alphas[2] + 0.00156892306269796 alphas[3] - alphas[4]
   + f_var[62] = 0
 R623: - 2094.959652373807 Fraction[62] + Q0_var[62] = 0
 R624: Q0_var[62] + Lost_0_expr_62 = 1645.146747751933
 R625: Q0_var[62] + Q_hat_adjusted_62 = 2078.080270222351
 R626: Sold_0[62] - Q0_var[62] - Q1_var[62] + total_stock_second_period_62
   = 0
 R627: total_stock_second_period_62 + Lost_1_expr_62 = 432.9229111357785
 R628: Sold_1[62] + Left_1[62] - total_stock_second_period_62 = 0
 R629: - 700 Sold_0[62] - 700 Sold_1[62] + 700 Lost_0[62] + 700 Lost_1[62]
   + 300 Left_1[62] + profits_vars[62] = 0
 Fraction_lower_bound_63: Fraction[63] >= 0
 Fraction_upper_bound_63: Fraction[63] <= 1
 R632: - 1.317272369956665 alphas[0] + 0.986116599337248 alphas[1]
   + 0.7372217393225176 alphas[2] + 0.2077022058405448 alphas[3]
   - alphas[4] + f_var[63] = 0
 R633: - 2094.959652373807 Fraction[63] + Q0_var[63] = 0
 R634: Q0_var[63] + Lost_0_expr_63 = 916.2209094918736
 R635: Q0_var[63] + Q_hat_adjusted_63 = 1151.496812051291
 R636: Sold_0[63] - Q0_var[63] - Q1_var[63] + total_stock_second_period_63
   = 0
 R637: total_stock_second_period_63 + Lost_1_expr_63 = 235.2652896022048
 R638: Sold_1[63] + Left_1[63] - total_stock_second_period_63 = 0
 R639: - 700 Sold_0[63] - 700 Sold_1[63] + 700 Lost_0[63] + 700 Lost_1[63]
   + 300 Left_1[63] + profits_vars[63] = 0
 Fraction_lower_bound_64: Fraction[64] >= 0
 Fraction_upper_bound_64: Fraction[64] <= 1
 R642: - 0.0595261692487954 alphas[0] - 0.0513809834646593 alphas[1]
   + 0.4333186017649998 alphas[2] - 1.327649784628208 alphas[3] - alphas[4]
   + f_var[64] = 0
 R643: - 2094.959652373807 Fraction[64] + Q0_var[64] = 0
 R644: Q0_var[64] + Lost_0_expr_64 = 1117.461520177727
 R645: Q0_var[64] + Q_hat_adjusted_64 = 1395.951444519438
 R646: Sold_0[64] - Q0_var[64] - Q1_var[64] + total_stock_second_period_64
   = 0
 R647: total_stock_second_period_64 + Lost_1_expr_64 = 278.4793111724015
 R648: Sold_1[64] + Left_1[64] - total_stock_second_period_64 = 0
 R649: - 700 Sold_0[64] - 700 Sold_1[64] + 700 Lost_0[64] + 700 Lost_1[64]
   + 300 Left_1[64] + profits_vars[64] = 0
 Fraction_lower_bound_65: Fraction[65] >= 0
 Fraction_upper_bound_65: Fraction[65] <= 1
 R652: - 1.489584777425652 alphas[0] - 0.8921842264297876 alphas[1]
   + 1.404423815040936 alphas[2] + 0.9289885905518318 alphas[3] - alphas[4]
   + f_var[65] = 0
 R653: - 2094.959652373807 Fraction[65] + Q0_var[65] = 0
 R654: Q0_var[65] + Lost_0_expr_65 = 587.4992360784864
 R655: Q0_var[65] + Q_hat_adjusted_65 = 727.9012112277599
 R656: Sold_0[65] - Q0_var[65] - Q1_var[65] + total_stock_second_period_65
   = 0
 R657: total_stock_second_period_65 + Lost_1_expr_65 = 140.3913599461545
 R658: Sold_1[65] + Left_1[65] - total_stock_second_period_65 = 0
 R659: - 700 Sold_0[65] - 700 Sold_1[65] + 700 Lost_0[65] + 700 Lost_1[65]
   + 300 Left_1[65] + profits_vars[65] = 0
 Fraction_lower_bound_66: Fraction[66] >= 0
 Fraction_upper_bound_66: Fraction[66] <= 1
 R662: - 0.1912767080796869 alphas[0] + 1.239397047654241 alphas[1]
   + 0.8105094300107493 alphas[2] + 1.202430081307104 alphas[3] - alphas[4]
   + f_var[66] = 0
 R663: - 2094.959652373807 Fraction[66] + Q0_var[66] = 0
 R664: Q0_var[66] + Lost_0_expr_66 = 1075.330732676315
 R665: Q0_var[66] + Q_hat_adjusted_66 = 1300.185369098844
 R666: Sold_0[66] - Q0_var[66] - Q1_var[66] + total_stock_second_period_66
   = 0
 R667: total_stock_second_period_66 + Lost_1_expr_66 = 224.8440222481231
 R668: Sold_1[66] + Left_1[66] - total_stock_second_period_66 = 0
 R669: - 700 Sold_0[66] - 700 Sold_1[66] + 700 Lost_0[66] + 700 Lost_1[66]
   + 300 Left_1[66] + profits_vars[66] = 0
 Fraction_lower_bound_67: Fraction[67] >= 0
 Fraction_upper_bound_67: Fraction[67] <= 1
 R672: - 1.492849864142074 alphas[0] - 1.604612761671799 alphas[1]
   + 0.5675946554482222 alphas[2] + 0.1472991989159017 alphas[3]
   - alphas[4] + f_var[67] = 0
 R673: - 2094.959652373807 Fraction[67] + Q0_var[67] = 0
 R674: Q0_var[67] + Lost_0_expr_67 = 1447.577073005654
 R675: Q0_var[67] + Q_hat_adjusted_67 = 1706.97338555129
 R676: Sold_0[67] - Q0_var[67] - Q1_var[67] + total_stock_second_period_67
   = 0
 R677: total_stock_second_period_67 + Lost_1_expr_67 = 259.3857002051752
 R678: Sold_1[67] + Left_1[67] - total_stock_second_period_67 = 0
 R679: - 700 Sold_0[67] - 700 Sold_1[67] + 700 Lost_0[67] + 700 Lost_1[67]
   + 300 Left_1[67] + profits_vars[67] = 0
 Fraction_lower_bound_68: Fraction[68] >= 0
 Fraction_upper_bound_68: Fraction[68] <= 1
 R682: - 1.174519908604492 alphas[0] + 0.1201720741930792 alphas[1]
   - 0.5687782594104671 alphas[2] + 0.1332876864023443 alphas[3]
   - alphas[4] + f_var[68] = 0
 R683: - 2094.959652373807 Fraction[68] + Q0_var[68] = 0
 R684: Q0_var[68] + Lost_0_expr_68 = 1464.106093317937
 R685: Q0_var[68] + Q_hat_adjusted_68 = 1885.090876989008
 R686: Sold_0[68] - Q0_var[68] - Q1_var[68] + total_stock_second_period_68
   = 0
 R687: total_stock_second_period_68 + Lost_1_expr_68 = 420.9741722728962
 R688: Sold_1[68] + Left_1[68] - total_stock_second_period_68 = 0
 R689: - 700 Sold_0[68] - 700 Sold_1[68] + 700 Lost_0[68] + 700 Lost_1[68]
   + 300 Left_1[68] + profits_vars[68] = 0
 Fraction_lower_bound_69: Fraction[69] >= 0
 Fraction_upper_bound_69: Fraction[69] <= 1
 R692: - 1.646139552440572 alphas[0] - 0.8952189343378403 alphas[1]
   + 1.081733420650266 alphas[2] + 0.6834068056039022 alphas[3] - alphas[4]
   + f_var[69] = 0
 R693: - 2094.959652373807 Fraction[69] + Q0_var[69] = 0
 R694: Q0_var[69] + Lost_0_expr_69 = 1046.814392873435
 R695: Q0_var[69] + Q_hat_adjusted_69 = 1233.10187499024
 R696: Sold_0[69] - Q0_var[69] - Q1_var[69] + total_stock_second_period_69
   = 0
 R697: total_stock_second_period_69 + Lost_1_expr_69 = 186.276867344065
 R698: Sold_1[69] + Left_1[69] - total_stock_second_period_69 = 0
 R699: - 700 Sold_0[69] - 700 Sold_1[69] + 700 Lost_0[69] + 700 Lost_1[69]
   + 300 Left_1[69] + profits_vars[69] = 0
 Fraction_lower_bound_70: Fraction[70] >= 0
 Fraction_upper_bound_70: Fraction[70] <= 1
 R702: - 1.480842863851214 alphas[0] + 0.5008806262643748 alphas[1]
   - 0.6926747924338338 alphas[2] - 1.13597896903422 alphas[3] - alphas[4]
   + f_var[70] = 0
 R703: - 2094.959652373807 Fraction[70] + Q0_var[70] = 0
 R704: Q0_var[70] + Lost_0_expr_70 = 1805.314202467221
 R705: Q0_var[70] + Q_hat_adjusted_70 = 2243.916663009344
 R706: Sold_0[70] - Q0_var[70] - Q1_var[70] + total_stock_second_period_70
   = 0
 R707: total_stock_second_period_70 + Lost_1_expr_70 = 438.5918497130675
 R708: Sold_1[70] + Left_1[70] - total_stock_second_period_70 = 0
 R709: - 700 Sold_0[70] - 700 Sold_1[70] + 700 Lost_0[70] + 700 Lost_1[70]
   + 300 Left_1[70] + profits_vars[70] = 0
 Fraction_lower_bound_71: Fraction[71] >= 0
 Fraction_upper_bound_71: Fraction[71] <= 1
 R712: 1.530908503101651 alphas[0] + 0.050383524126628 alphas[1]
   + 1.274791280266404 alphas[2] - 1.166760235393025 alphas[3] - alphas[4]
   + f_var[71] = 0
 R713: - 2094.959652373807 Fraction[71] + Q0_var[71] = 0
 R714: Q0_var[71] + Lost_0_expr_71 = 1027.159669445558
 R715: Q0_var[71] + Q_hat_adjusted_71 = 1185.994961699985
 R716: Sold_0[71] - Q0_var[71] - Q1_var[71] + total_stock_second_period_71
   = 0
 R717: total_stock_second_period_71 + Lost_1_expr_71 = 158.8246778836932
 R718: Sold_1[71] + Left_1[71] - total_stock_second_period_71 = 0
 R719: - 700 Sold_0[71] - 700 Sold_1[71] + 700 Lost_0[71] + 700 Lost_1[71]
   + 300 Left_1[71] + profits_vars[71] = 0
 Fraction_lower_bound_72: Fraction[72] >= 0
 Fraction_upper_bound_72: Fraction[72] <= 1
 R722: 1.058098998724101 alphas[0] - 1.532018911042912 alphas[1]
   - 1.455499878579862 alphas[2] - 0.7157639337969846 alphas[3] - alphas[4]
   + f_var[72] = 0
 R723: - 2094.959652373807 Fraction[72] + Q0_var[72] = 0
 R724: Q0_var[72] + Lost_0_expr_72 = 2026.589469105
 R725: Q0_var[72] + Q_hat_adjusted_72 = 2573.663134228191
 R726: Sold_0[72] - Q0_var[72] - Q1_var[72] + total_stock_second_period_72
   = 0
 R727: total_stock_second_period_72 + Lost_1_expr_72 = 547.0630553476674
 R728: Sold_1[72] + Left_1[72] - total_stock_second_period_72 = 0
 R729: - 700 Sold_0[72] - 700 Sold_1[72] + 700 Lost_0[72] + 700 Lost_1[72]
   + 300 Left_1[72] + profits_vars[72] = 0
 Fraction_lower_bound_73: Fraction[73] >= 0
 Fraction_upper_bound_73: Fraction[73] <= 1
 R732: 0.3275668248718018 alphas[0] - 0.6205316554417434 alphas[1]
   - 0.5142348430083187 alphas[2] - 1.63443895218256 alphas[3] - alphas[4]
   + f_var[73] = 0
 R733: - 2094.959652373807 Fraction[73] + Q0_var[73] = 0
 R734: Q0_var[73] + Lost_0_expr_73 = 1653.272965709546
 R735: Q0_var[73] + Q_hat_adjusted_73 = 2066.501856645955
 R736: Sold_0[73] - Q0_var[73] - Q1_var[73] + total_stock_second_period_73
   = 0
 R737: total_stock_second_period_73 + Lost_1_expr_73 = 413.2182787607123
 R738: Sold_1[73] + Left_1[73] - total_stock_second_period_73 = 0
 R739: - 700 Sold_0[73] - 700 Sold_1[73] + 700 Lost_0[73] + 700 Lost_1[73]
   + 300 Left_1[73] + profits_vars[73] = 0
 Fraction_lower_bound_74: Fraction[74] >= 0
 Fraction_upper_bound_74: Fraction[74] <= 1
 R742: - 1.591122969612255 alphas[0] + 1.585506134986335 alphas[1]
   - 0.3420650972620996 alphas[2] + 1.417670723960079 alphas[3] - alphas[4]
   + f_var[74] = 0
 R743: - 2094.959652373807 Fraction[74] + Q0_var[74] = 0
 R744: Q0_var[74] + Lost_0_expr_74 = 1507.986653427202
 R745: Q0_var[74] + Q_hat_adjusted_74 = 1896.733575348264
 R746: Sold_0[74] - Q0_var[74] - Q1_var[74] + total_stock_second_period_74
   = 0
 R747: total_stock_second_period_74 + Lost_1_expr_74 = 388.7363100445972
 R748: Sold_1[74] + Left_1[74] - total_stock_second_period_74 = 0
 R749: - 700 Sold_0[74] - 700 Sold_1[74] + 700 Lost_0[74] + 700 Lost_1[74]
   + 300 Left_1[74] + profits_vars[74] = 0
 Fraction_lower_bound_75: Fraction[75] >= 0
 Fraction_upper_bound_75: Fraction[75] <= 1
 R752: 0.6312326726982076 alphas[0] + 0.9006381553950108 alphas[1]
   + 1.029981320696598 alphas[2] - 1.607940050101634 alphas[3] - alphas[4]
   + f_var[75] = 0
 R753: - 2094.959652373807 Fraction[75] + Q0_var[75] = 0
 R754: Q0_var[75] + Lost_0_expr_75 = 854.2668527783042
 R755: Q0_var[75] + Q_hat_adjusted_75 = 1047.913310729251
 R756: Sold_0[75] - Q0_var[75] - Q1_var[75] + total_stock_second_period_75
   = 0
 R757: total_stock_second_period_75 + Lost_1_expr_75 = 193.6358448848518
 R758: Sold_1[75] + Left_1[75] - total_stock_second_period_75 = 0
 R759: - 700 Sold_0[75] - 700 Sold_1[75] + 700 Lost_0[75] + 700 Lost_1[75]
   + 300 Left_1[75] + profits_vars[75] = 0
 Fraction_lower_bound_76: Fraction[76] >= 0
 Fraction_upper_bound_76: Fraction[76] <= 1
 R762: 1.106847203169945 alphas[0] + 0.8100921647695145 alphas[1]
   - 0.9166144260118826 alphas[2] + 1.19559803631902 alphas[3] - alphas[4]
   + f_var[76] = 0
 R763: - 2094.959652373807 Fraction[76] + Q0_var[76] = 0
 R764: Q0_var[76] + Lost_0_expr_76 = 1802.793700748482
 R765: Q0_var[76] + Q_hat_adjusted_76 = 2273.239636999629
 R766: Sold_0[76] - Q0_var[76] - Q1_var[76] + total_stock_second_period_76
   = 0
 R767: total_stock_second_period_76 + Lost_1_expr_76 = 470.4353253469187
 R768: Sold_1[76] + Left_1[76] - total_stock_second_period_76 = 0
 R769: - 700 Sold_0[76] - 700 Sold_1[76] + 700 Lost_0[76] + 700 Lost_1[76]
   + 300 Left_1[76] + profits_vars[76] = 0
 Fraction_lower_bound_77: Fraction[77] >= 0
 Fraction_upper_bound_77: Fraction[77] <= 1
 R772: - 1.366751816736344 alphas[0] - 1.51224477649207 alphas[1]
   + 0.8584566860099672 alphas[2] - 0.8249352172290247 alphas[3]
   - alphas[4] + f_var[77] = 0
 R773: - 2094.959652373807 Fraction[77] + Q0_var[77] = 0
 R774: Q0_var[77] + Lost_0_expr_77 = 886.701064888008
 R775: Q0_var[77] + Q_hat_adjusted_77 = 1104.737760836166
 R776: Sold_0[77] - Q0_var[77] - Q1_var[77] + total_stock_second_period_77
   = 0
 R777: total_stock_second_period_77 + Lost_1_expr_77 = 218.0260809407732
 R778: Sold_1[77] + Left_1[77] - total_stock_second_period_77 = 0
 R779: - 700 Sold_0[77] - 700 Sold_1[77] + 700 Lost_0[77] + 700 Lost_1[77]
   + 300 Left_1[77] + profits_vars[77] = 0
 Fraction_lower_bound_78: Fraction[78] >= 0
 Fraction_upper_bound_78: Fraction[78] <= 1
 R782: 0.1324298610493315 alphas[0] - 0.883773924404215 alphas[1]
   - 1.117647827450436 alphas[2] + 0.1608215394317186 alphas[3] - alphas[4]
   + f_var[78] = 0
 R783: - 2094.959652373807 Fraction[78] + Q0_var[78] = 0
 R784: Q0_var[78] + Lost_0_expr_78 = 2390.950436595269
 R785: Q0_var[78] + Q_hat_adjusted_78 = 2889.982655455022
 R786: Sold_0[78] - Q0_var[78] - Q1_var[78] + total_stock_second_period_78
   = 0
 R787: total_stock_second_period_78 + Lost_1_expr_78 = 499.0216104899979
 R788: Sold_1[78] + Left_1[78] - total_stock_second_period_78 = 0
 R789: - 700 Sold_0[78] - 700 Sold_1[78] + 700 Lost_0[78] + 700 Lost_1[78]
   + 300 Left_1[78] + profits_vars[78] = 0
 Fraction_lower_bound_79: Fraction[79] >= 0
 Fraction_upper_bound_79: Fraction[79] <= 1
 R792: - 1.440263948784893 alphas[0] - 0.5819230614172639 alphas[1]
   - 1.050527301527524 alphas[2] - 0.123609708384353 alphas[3] - alphas[4]
   + f_var[79] = 0
 R793: - 2094.959652373807 Fraction[79] + Q0_var[79] = 0
 R794: Q0_var[79] + Lost_0_expr_79 = 2161.675742814894
 R795: Q0_var[79] + Q_hat_adjusted_79 = 2651.163648894815
 R796: Sold_0[79] - Q0_var[79] - Q1_var[79] + total_stock_second_period_79
   = 0
 R797: total_stock_second_period_79 + Lost_1_expr_79 = 489.4772935791971
 R798: Sold_1[79] + Left_1[79] - total_stock_second_period_79 = 0
 R799: - 700 Sold_0[79] - 700 Sold_1[79] + 700 Lost_0[79] + 700 Lost_1[79]
   + 300 Left_1[79] + profits_vars[79] = 0
 Fraction_lower_bound_80: Fraction[80] >= 0
 Fraction_upper_bound_80: Fraction[80] <= 1
 R802: 1.107655486821045 alphas[0] + 1.439465082647801 alphas[1]
   - 1.614679535735436 alphas[2] + 0.3865636081022111 alphas[3] - alphas[4]
   + f_var[80] = 0
 R803: - 2094.959652373807 Fraction[80] + Q0_var[80] = 0
 R804: Q0_var[80] + Lost_0_expr_80 = 2503.987807721447
 R805: Q0_var[80] + Q_hat_adjusted_80 = 3073.696294174896
 R806: Sold_0[80] - Q0_var[80] - Q1_var[80] + total_stock_second_period_80
   = 0
 R807: total_stock_second_period_80 + Lost_1_expr_80 = 569.6978764065832
 R808: Sold_1[80] + Left_1[80] - total_stock_second_period_80 = 0
 R809: - 700 Sold_0[80] - 700 Sold_1[80] + 700 Lost_0[80] + 700 Lost_1[80]
   + 300 Left_1[80] + profits_vars[80] = 0
 Fraction_lower_bound_81: Fraction[81] >= 0
 Fraction_upper_bound_81: Fraction[81] <= 1
 R812: - 0.5993306080836944 alphas[0] - 0.9974462616401637 alphas[1]
   - 1.063673332770626 alphas[2] + 0.4709371606980376 alphas[3] - alphas[4]
   + f_var[81] = 0
 R813: - 2094.959652373807 Fraction[81] + Q0_var[81] = 0
 R814: Q0_var[81] + Lost_0_expr_81 = 2277.635595132306
 R815: Q0_var[81] + Q_hat_adjusted_81 = 2768.992823517321
 R816: Sold_0[81] - Q0_var[81] - Q1_var[81] + total_stock_second_period_81
   = 0
 R817: total_stock_second_period_81 + Lost_1_expr_81 = 491.3466157660876
 R818: Sold_1[81] + Left_1[81] - total_stock_second_period_81 = 0
 R819: - 700 Sold_0[81] - 700 Sold_1[81] + 700 Lost_0[81] + 700 Lost_1[81]
   + 300 Left_1[81] + profits_vars[81] = 0
 Fraction_lower_bound_82: Fraction[82] >= 0
 Fraction_upper_bound_82: Fraction[82] <= 1
 R822: 0.1533227070571766 alphas[0] + 1.06762849568638 alphas[1]
   + 0.4052358005765777 alphas[2] - 0.3942227834589006 alphas[3]
   - alphas[4] + f_var[82] = 0
 R823: - 2094.959652373807 Fraction[82] + Q0_var[82] = 0
 R824: Q0_var[82] + Lost_0_expr_82 = 1087.845412437135
 R825: Q0_var[82] + Q_hat_adjusted_82 = 1370.328617488549
 R826: Sold_0[82] - Q0_var[82] - Q1_var[82] + total_stock_second_period_82
   = 0
 R827: total_stock_second_period_82 + Lost_1_expr_82 = 282.4725926701232
 R828: Sold_1[82] + Left_1[82] - total_stock_second_period_82 = 0
 R829: - 700 Sold_0[82] - 700 Sold_1[82] + 700 Lost_0[82] + 700 Lost_1[82]
   + 300 Left_1[82] + profits_vars[82] = 0
 Fraction_lower_bound_83: Fraction[83] >= 0
 Fraction_upper_bound_83: Fraction[83] <= 1
 R832: 1.393047828633475 alphas[0] + 1.176809590040238 alphas[1]
   - 0.1071075477747847 alphas[2] - 0.3740830844408593 alphas[3]
   - alphas[4] + f_var[83] = 0
 R833: - 2094.959652373807 Fraction[83] + Q0_var[83] = 0
 R834: Q0_var[83] + Lost_0_expr_83 = 1543.614078096632
 R835: Q0_var[83] + Q_hat_adjusted_83 = 1898.950813272298
 R836: Sold_0[83] - Q0_var[83] - Q1_var[83] + total_stock_second_period_83
   = 0
 R837: total_stock_second_period_83 + Lost_1_expr_83 = 355.326123189571
 R838: Sold_1[83] + Left_1[83] - total_stock_second_period_83 = 0
 R839: - 700 Sold_0[83] - 700 Sold_1[83] + 700 Lost_0[83] + 700 Lost_1[83]
   + 300 Left_1[83] + profits_vars[83] = 0
 Fraction_lower_bound_84: Fraction[84] >= 0
 Fraction_upper_bound_84: Fraction[84] <= 1
 R842: - 0.7197859180637236 alphas[0] - 1.213318324533186 alphas[1]
   - 0.5157555663787907 alphas[2] - 0.7889794104400278 alphas[3]
   - alphas[4] + f_var[84] = 0
 R843: - 2094.959652373807 Fraction[84] + Q0_var[84] = 0
 R844: Q0_var[84] + Lost_0_expr_84 = 1996.724084129629
 R845: Q0_var[84] + Q_hat_adjusted_84 = 2410.169216114634
 R846: Sold_0[84] - Q0_var[84] - Q1_var[84] + total_stock_second_period_84
   = 0
 R847: total_stock_second_period_84 + Lost_1_expr_84 = 413.4345205970489
 R848: Sold_1[84] + Left_1[84] - total_stock_second_period_84 = 0
 R849: - 700 Sold_0[84] - 700 Sold_1[84] + 700 Lost_0[84] + 700 Lost_1[84]
   + 300 Left_1[84] + profits_vars[84] = 0
 Fraction_lower_bound_85: Fraction[85] >= 0
 Fraction_upper_bound_85: Fraction[85] <= 1
 R852: 0.8105923883780601 alphas[0] + 1.653937118460339 alphas[1]
   + 0.7478145107826256 alphas[2] - 0.7990610448866067 alphas[3]
   - alphas[4] + f_var[85] = 0
 R853: - 2094.959652373807 Fraction[85] + Q0_var[85] = 0
 R854: Q0_var[85] + Lost_0_expr_85 = 947.280030676051
 R855: Q0_var[85] + Q_hat_adjusted_85 = 1181.049677166554
 R856: Sold_0[85] - Q0_var[85] - Q1_var[85] + total_stock_second_period_85
   = 0
 R857: total_stock_second_period_85 + Lost_1_expr_85 = 233.759032516331
 R858: Sold_1[85] + Left_1[85] - total_stock_second_period_85 = 0
 R859: - 700 Sold_0[85] - 700 Sold_1[85] + 700 Lost_0[85] + 700 Lost_1[85]
   + 300 Left_1[85] + profits_vars[85] = 0
 Fraction_lower_bound_86: Fraction[86] >= 0
 Fraction_upper_bound_86: Fraction[86] <= 1
 R862: 1.51870231620292 alphas[0] + 1.460611873093213 alphas[1]
   - 1.124709866741962 alphas[2] + 1.631386859757011 alphas[3] - alphas[4]
   + f_var[86] = 0
 R863: - 2094.959652373807 Fraction[86] + Q0_var[86] = 0
 R864: Q0_var[86] + Lost_0_expr_86 = 2065.909526868893
 R865: Q0_var[86] + Q_hat_adjusted_86 = 2565.945945994979
 R866: Sold_0[86] - Q0_var[86] - Q1_var[86] + total_stock_second_period_86
   = 0
 R867: total_stock_second_period_86 + Lost_1_expr_86 = 500.0258091327837
 R868: Sold_1[86] + Left_1[86] - total_stock_second_period_86 = 0
 R869: - 700 Sold_0[86] - 700 Sold_1[86] + 700 Lost_0[86] + 700 Lost_1[86]
   + 300 Left_1[86] + profits_vars[86] = 0
 Fraction_lower_bound_87: Fraction[87] >= 0
 Fraction_upper_bound_87: Fraction[87] <= 1
 R872: 1.44942334856222 alphas[0] - 1.381129240638397 alphas[1]
   + 0.5312835373196112 alphas[2] - 0.1806537465885662 alphas[3]
   - alphas[4] + f_var[87] = 0
 R873: - 2094.959652373807 Fraction[87] + Q0_var[87] = 0
 R874: Q0_var[87] + Lost_0_expr_87 = 964.8857025058114
 R875: Q0_var[87] + Q_hat_adjusted_87 = 1229.445336656935
 R876: Sold_0[87] - Q0_var[87] - Q1_var[87] + total_stock_second_period_87
   = 0
 R877: total_stock_second_period_87 + Lost_1_expr_87 = 264.5490211721242
 R878: Sold_1[87] + Left_1[87] - total_stock_second_period_87 = 0
 R879: - 700 Sold_0[87] - 700 Sold_1[87] + 700 Lost_0[87] + 700 Lost_1[87]
   + 300 Left_1[87] + profits_vars[87] = 0
 Fraction_lower_bound_88: Fraction[88] >= 0
 Fraction_upper_bound_88: Fraction[88] <= 1
 R882: 1.4455579463565 alphas[0] + 0.8356684711768414 alphas[1]
   + 1.230884766141699 alphas[2] - 0.6192833728680207 alphas[3] - alphas[4]
   + f_var[88] = 0
 R883: - 2094.959652373807 Fraction[88] + Q0_var[88] = 0
 R884: Q0_var[88] + Lost_0_expr_88 = 776.1424780634057
 R885: Q0_var[88] + Q_hat_adjusted_88 = 941.2211320568795
 R886: Sold_0[88] - Q0_var[88] - Q1_var[88] + total_stock_second_period_88
   = 0
 R887: total_stock_second_period_88 + Lost_1_expr_88 = 165.0680390544581
 R888: Sold_1[88] + Left_1[88] - total_stock_second_period_88 = 0
 R889: - 700 Sold_0[88] - 700 Sold_1[88] + 700 Lost_0[88] + 700 Lost_1[88]
   + 300 Left_1[88] + profits_vars[88] = 0
 Fraction_lower_bound_89: Fraction[89] >= 0
 Fraction_upper_bound_89: Fraction[89] <= 1
 R892: - 1.439777852858013 alphas[0] - 0.8996056819157069 alphas[1]
   + 1.479183962308467 alphas[2] - 1.020364849079581 alphas[3] - alphas[4]
   + f_var[89] = 0
 R893: - 2094.959652373807 Fraction[89] + Q0_var[89] = 0
 R894: Q0_var[89] + Lost_0_expr_89 = 421.9826301279803
 R895: Q0_var[89] + Q_hat_adjusted_89 = 551.753959683921
 R896: Sold_0[89] - Q0_var[89] - Q1_var[89] + total_stock_second_period_89
   = 0
 R897: total_stock_second_period_89 + Lost_1_expr_89 = 129.7607142723991
 R898: Sold_1[89] + Left_1[89] - total_stock_second_period_89 = 0
 R899: - 700 Sold_0[89] - 700 Sold_1[89] + 700 Lost_0[89] + 700 Lost_1[89]
   + 300 Left_1[89] + profits_vars[89] = 0
 Fraction_lower_bound_90: Fraction[90] >= 0
 Fraction_upper_bound_90: Fraction[90] <= 1
 R902: - 0.8677607094953067 alphas[0] - 0.5106993715123209 alphas[1]
   - 0.6322854897127962 alphas[2] - 1.407816429597476 alphas[3] - alphas[4]
   + f_var[90] = 0
 R903: - 2094.959652373807 Fraction[90] + Q0_var[90] = 0
 R904: Q0_var[90] + Lost_0_expr_90 = 1723.416478483673
 R905: Q0_var[90] + Q_hat_adjusted_90 = 2153.431779528945
 R906: Sold_0[90] - Q0_var[90] - Q1_var[90] + total_stock_second_period_90
   = 0
 R907: total_stock_second_period_90 + Lost_1_expr_90 = 430.0046904906585
 R908: Sold_1[90] + Left_1[90] - total_stock_second_period_90 = 0
 R909: - 700 Sold_0[90] - 700 Sold_1[90] + 700 Lost_0[90] + 700 Lost_1[90]
   + 300 Left_1[90] + profits_vars[90] = 0
 Fraction_lower_bound_91: Fraction[91] >= 0
 Fraction_upper_bound_91: Fraction[91] <= 1
 R912: - 1.40682116062353 alphas[0] + 1.298123813096012 alphas[1]
   + 0.7576835723249721 alphas[2] + 0.5830791675421132 alphas[3]
   - alphas[4] + f_var[91] = 0
 R913: - 2094.959652373807 Fraction[91] + Q0_var[91] = 0
 R914: Q0_var[91] + Lost_0_expr_91 = 540.6256937009452
 R915: Q0_var[91] + Q_hat_adjusted_91 = 772.9919908116011
 R916: Sold_0[91] - Q0_var[91] - Q1_var[91] + total_stock_second_period_91
   = 0
 R917: total_stock_second_period_91 + Lost_1_expr_91 = 232.3556845884478
 R918: Sold_1[91] + Left_1[91] - total_stock_second_period_91 = 0
 R919: - 700 Sold_0[91] - 700 Sold_1[91] + 700 Lost_0[91] + 700 Lost_1[91]
   + 300 Left_1[91] + profits_vars[91] = 0
 Fraction_lower_bound_92: Fraction[92] >= 0
 Fraction_upper_bound_92: Fraction[92] <= 1
 R922: - 0.6384017998977266 alphas[0] - 0.9679521983998034 alphas[1]
   - 1.30588173004993 alphas[2] + 0.3431827630909145 alphas[3] - alphas[4]
   + f_var[92] = 0
 R923: - 2094.959652373807 Fraction[92] + Q0_var[92] = 0
 R924: Q0_var[92] + Lost_0_expr_92 = 2256.509254840888
 R925: Q0_var[92] + Q_hat_adjusted_92 = 2782.307715215979
 R926: Sold_0[92] - Q0_var[92] - Q1_var[92] + total_stock_second_period_92
   = 0
 R927: total_stock_second_period_92 + Lost_1_expr_92 = 525.7878492085347
 R928: Sold_1[92] + Left_1[92] - total_stock_second_period_92 = 0
 R929: - 700 Sold_0[92] - 700 Sold_1[92] + 700 Lost_0[92] + 700 Lost_1[92]
   + 300 Left_1[92] + profits_vars[92] = 0
 Fraction_lower_bound_93: Fraction[93] >= 0
 Fraction_upper_bound_93: Fraction[93] <= 1
 R932: - 0.1488791589709158 alphas[0] + 0.9822636020645221 alphas[1]
   - 1.02759168188635 alphas[2] + 1.386819566363409 alphas[3] - alphas[4]
   + f_var[93] = 0
 R933: - 2094.959652373807 Fraction[93] + Q0_var[93] = 0
 R934: Q0_var[93] + Lost_0_expr_93 = 2186.369116771933
 R935: Q0_var[93] + Q_hat_adjusted_93 = 2672.595652456549
 R936: Sold_0[93] - Q0_var[93] - Q1_var[93] + total_stock_second_period_93
   = 0
 R937: total_stock_second_period_93 + Lost_1_expr_93 = 486.2159242828306
 R938: Sold_1[93] + Left_1[93] - total_stock_second_period_93 = 0
 R939: - 700 Sold_0[93] - 700 Sold_1[93] + 700 Lost_0[93] + 700 Lost_1[93]
   + 300 Left_1[93] + profits_vars[93] = 0
 Fraction_lower_bound_94: Fraction[94] >= 0
 Fraction_upper_bound_94: Fraction[94] <= 1
 R942: 0.6161783464923033 alphas[0] - 0.6376739384269547 alphas[1]
   + 1.283375883095995 alphas[2] + 0.1636389119382188 alphas[3] - alphas[4]
   + f_var[94] = 0
 R943: - 2094.959652373807 Fraction[94] + Q0_var[94] = 0
 R944: Q0_var[94] + Lost_0_expr_94 = 729.9747158876983
 R945: Q0_var[94] + Q_hat_adjusted_94 = 887.5893063721257
 R946: Sold_0[94] - Q0_var[94] - Q1_var[94] + total_stock_second_period_94
   = 0
 R947: total_stock_second_period_94 + Lost_1_expr_94 = 157.6039757357379
 R948: Sold_1[94] + Left_1[94] - total_stock_second_period_94 = 0
 R949: - 700 Sold_0[94] - 700 Sold_1[94] + 700 Lost_0[94] + 700 Lost_1[94]
   + 300 Left_1[94] + profits_vars[94] = 0
 Fraction_lower_bound_95: Fraction[95] >= 0
 Fraction_upper_bound_95: Fraction[95] <= 1
 R952: - 1.747272629222934 alphas[0] + 0.4064896491446977 alphas[1]
   - 0.2640344785257809 alphas[2] - 1.503879824954055 alphas[3] - alphas[4]
   + f_var[95] = 0
 R953: - 2094.959652373807 Fraction[95] + Q0_var[95] = 0
 R954: Q0_var[95] + Lost_0_expr_95 = 1588.715952569364
 R955: Q0_var[95] + Q_hat_adjusted_95 = 1966.367178197391
 R956: Sold_0[95] - Q0_var[95] - Q1_var[95] + total_stock_second_period_95
   = 0
 R957: total_stock_second_period_95 + Lost_1_expr_95 = 377.6406140004296
 R958: Sold_1[95] + Left_1[95] - total_stock_second_period_95 = 0
 R959: - 700 Sold_0[95] - 700 Sold_1[95] + 700 Lost_0[95] + 700 Lost_1[95]
   + 300 Left_1[95] + profits_vars[95] = 0
 Fraction_lower_bound_96: Fraction[96] >= 0
 Fraction_upper_bound_96: Fraction[96] <= 1
 R962: 0.4193955035383451 alphas[0] - 0.137565580496342 alphas[1]
   - 1.713057853628026 alphas[2] + 0.6919816637357157 alphas[3] - alphas[4]
   + f_var[96] = 0
 R963: - 2094.959652373807 Fraction[96] + Q0_var[96] = 0
 R964: Q0_var[96] + Lost_0_expr_96 = 2584.39013815203
 R965: Q0_var[96] + Q_hat_adjusted_96 = 3168.087693233727
 R966: Sold_0[96] - Q0_var[96] - Q1_var[96] + total_stock_second_period_96
   = 0
 R967: total_stock_second_period_96 + Lost_1_expr_96 = 583.6869480088644
 R968: Sold_1[96] + Left_1[96] - total_stock_second_period_96 = 0
 R969: - 700 Sold_0[96] - 700 Sold_1[96] + 700 Lost_0[96] + 700 Lost_1[96]
   + 300 Left_1[96] + profits_vars[96] = 0
 Fraction_lower_bound_97: Fraction[97] >= 0
 Fraction_upper_bound_97: Fraction[97] <= 1
 R972: 0.0497721995776152 alphas[0] - 1.470088097483792 alphas[1]
   - 1.227311783792568 alphas[2] + 0.8942495932048201 alphas[3] - alphas[4]
   + f_var[97] = 0
 R973: - 2094.959652373807 Fraction[97] + Q0_var[97] = 0
 R974: Q0_var[97] + Lost_0_expr_97 = 2206.945975697695
 R975: Q0_var[97] + Q_hat_adjusted_97 = 2721.572049643619
 R976: Sold_0[97] - Q0_var[97] - Q1_var[97] + total_stock_second_period_97
   = 0
 R977: total_stock_second_period_97 + Lost_1_expr_97 = 514.6154625736905
 R978: Sold_1[97] + Left_1[97] - total_stock_second_period_97 = 0
 R979: - 700 Sold_0[97] - 700 Sold_1[97] + 700 Lost_0[97] + 700 Lost_1[97]
   + 300 Left_1[97] + profits_vars[97] = 0
 Fraction_lower_bound_98: Fraction[98] >= 0
 Fraction_upper_bound_98: Fraction[98] <= 1
 R982: 0.3646774448736319 alphas[0] + 0.8103605854717109 alphas[1]
   - 1.378541175127705 alphas[2] + 1.152256477544582 alphas[3] - alphas[4]
   + f_var[98] = 0
 R983: - 2094.959652373807 Fraction[98] + Q0_var[98] = 0
 R984: Q0_var[98] + Lost_0_expr_98 = 2382.086117672011
 R985: Q0_var[98] + Q_hat_adjusted_98 = 2918.216509912859
 R986: Sold_0[98] - Q0_var[98] - Q1_var[98] + total_stock_second_period_98
   = 0
 R987: total_stock_second_period_98 + Lost_1_expr_98 = 536.1197821135552
 R988: Sold_1[98] + Left_1[98] - total_stock_second_period_98 = 0
 R989: - 700 Sold_0[98] - 700 Sold_1[98] + 700 Lost_0[98] + 700 Lost_1[98]
   + 300 Left_1[98] + profits_vars[98] = 0
 Fraction_lower_bound_99: Fraction[99] >= 0
 Fraction_upper_bound_99: Fraction[99] <= 1
 R992: - 1.684446412015219 alphas[0] + 0.6712168819921139 alphas[1]
   + 0.7645566914362433 alphas[2] - 0.3176750595252491 alphas[3]
   - alphas[4] + f_var[99] = 0
 R993: - 2094.959652373807 Fraction[99] + Q0_var[99] = 0
 R994: Q0_var[99] + Lost_0_expr_99 = 696.2362741724085
 R995: Q0_var[99] + Q_hat_adjusted_99 = 927.6252371071917
 R996: Sold_0[99] - Q0_var[99] - Q1_var[99] + total_stock_second_period_99
   = 0
 R997: total_stock_second_period_99 + Lost_1_expr_99 = 231.3783497707947
 R998: Sold_1[99] + Left_1[99] - total_stock_second_period_99 = 0
 R999: - 700 Sold_0[99] - 700 Sold_1[99] + 700 Lost_0[99] + 700 Lost_1[99]
   + 300 Left_1[99] + profits_vars[99] = 0
 Fraction_lower_bound_100: Fraction[100] >= 0
 Fraction_upper_bound_100: Fraction[100] <= 1
 R1002: 1.058486203095691 alphas[0] + 0.6996767124171692 alphas[1]
   - 0.8624913621974192 alphas[2] - 0.1154233847943529 alphas[3]
   - alphas[4] + f_var[100] = 0
 R1003: - 2094.959652373807 Fraction[100] + Q0_var[100] = 0
 R1004: Q0_var[100] + Lost_0_expr_100 = 1871.557184261849
 R1005: Q0_var[100] + Q_hat_adjusted_100 = 2334.306999240401
 R1006: Sold_0[100] - Q0_var[100] - Q1_var[100]
   + total_stock_second_period_100 = 0
 R1007: total_stock_second_period_100 + Lost_1_expr_100
   = 462.7392045831747
 R1008: Sold_1[100] + Left_1[100] - total_stock_second_period_100 = 0
 R1009: - 700 Sold_0[100] - 700 Sold_1[100] + 700 Lost_0[100]
   + 700 Lost_1[100] + 300 Left_1[100] + profits_vars[100] = 0
 Fraction_lower_bound_101: Fraction[101] >= 0
 Fraction_upper_bound_101: Fraction[101] <= 1
 R1012: 0.5359501859789113 alphas[0] + 0.4424397330442172 alphas[1]
   + 0.4933655967664428 alphas[2] + 0.1554006901208625 alphas[3]
   - alphas[4] + f_var[101] = 0
 R1013: - 2094.959652373807 Fraction[101] + Q0_var[101] = 0
 R1014: Q0_var[101] + Lost_0_expr_101 = 992.2571077137886
 R1015: Q0_var[101] + Q_hat_adjusted_101 = 1262.208549107395
 R1016: Sold_0[101] - Q0_var[101] - Q1_var[101]
   + total_stock_second_period_101 = 0
 R1017: total_stock_second_period_101 + Lost_1_expr_101
   = 269.9408269761772
 R1018: Sold_1[101] + Left_1[101] - total_stock_second_period_101 = 0
 R1019: - 700 Sold_0[101] - 700 Sold_1[101] + 700 Lost_0[101]
   + 700 Lost_1[101] + 300 Left_1[101] + profits_vars[101] = 0
 Fraction_lower_bound_102: Fraction[102] >= 0
 Fraction_upper_bound_102: Fraction[102] <= 1
 R1022: - 0.6648689637820886 alphas[0] - 0.2433304227868646 alphas[1]
   - 1.223203873734244 alphas[2] + 0.7901850226938798 alphas[3] - alphas[4]
   + f_var[102] = 0
 R1023: - 2094.959652373807 Fraction[102] + Q0_var[102] = 0
 R1024: Q0_var[102] + Lost_0_expr_102 = 2134.903060812388
 R1025: Q0_var[102] + Q_hat_adjusted_102 = 2648.945001393331
 R1026: Sold_0[102] - Q0_var[102] - Q1_var[102]
   + total_stock_second_period_102 = 0
 R1027: total_stock_second_period_102 + Lost_1_expr_102
   = 514.0313313426161
 R1028: Sold_1[102] + Left_1[102] - total_stock_second_period_102 = 0
 R1029: - 700 Sold_0[102] - 700 Sold_1[102] + 700 Lost_0[102]
   + 700 Lost_1[102] + 300 Left_1[102] + profits_vars[102] = 0
 Fraction_lower_bound_103: Fraction[103] >= 0
 Fraction_upper_bound_103: Fraction[103] <= 1
 R1032: 1.438302194464185 alphas[0] - 0.9856184842253388 alphas[1]
   + 1.327525063475418 alphas[2] - 1.721861940700691 alphas[3] - alphas[4]
   + f_var[103] = 0
 R1033: - 2094.959652373807 Fraction[103] + Q0_var[103] = 0
 R1034: Q0_var[103] + Lost_0_expr_103 = 265.183037246099
 R1035: Q0_var[103] + Q_hat_adjusted_103 = 416.5197597397508
 R1036: Sold_0[103] - Q0_var[103] - Q1_var[103]
   + total_stock_second_period_103 = 0
 R1037: total_stock_second_period_103 + Lost_1_expr_103
   = 151.3261088659059
 R1038: Sold_1[103] + Left_1[103] - total_stock_second_period_103 = 0
 R1039: - 700 Sold_0[103] - 700 Sold_1[103] + 700 Lost_0[103]
   + 700 Lost_1[103] + 300 Left_1[103] + profits_vars[103] = 0
 Fraction_lower_bound_104: Fraction[104] >= 0
 Fraction_upper_bound_104: Fraction[104] <= 1
 R1042: - 0.415751969848925 alphas[0] - 1.29385658919387 alphas[1]
   - 0.7205601061901921 alphas[2] + 1.167404206654798 alphas[3] - alphas[4]
   + f_var[104] = 0
 R1043: - 2094.959652373807 Fraction[104] + Q0_var[104] = 0
 R1044: Q0_var[104] + Lost_0_expr_104 = 1706.956062350333
 R1045: Q0_var[104] + Q_hat_adjusted_104 = 2149.523722751184
 R1046: Sold_0[104] - Q0_var[104] - Q1_var[104]
   + total_stock_second_period_104 = 0
 R1047: total_stock_second_period_104 + Lost_1_expr_104
   = 442.5570491507696
 R1048: Sold_1[104] + Left_1[104] - total_stock_second_period_104 = 0
 R1049: - 700 Sold_0[104] - 700 Sold_1[104] + 700 Lost_0[104]
   + 700 Lost_1[104] + 300 Left_1[104] + profits_vars[104] = 0
 Fraction_lower_bound_105: Fraction[105] >= 0
 Fraction_upper_bound_105: Fraction[105] <= 1
 R1052: 0.0259250529704071 alphas[0] + 1.467552533260718 alphas[1]
   - 0.5791722249120379 alphas[2] + 0.756557713004496 alphas[3] - alphas[4]
   + f_var[105] = 0
 R1053: - 2094.959652373807 Fraction[105] + Q0_var[105] = 0
 R1054: Q0_var[105] + Lost_0_expr_105 = 1584.403187941445
 R1055: Q0_var[105] + Q_hat_adjusted_105 = 2006.86595915796
 R1056: Sold_0[105] - Q0_var[105] - Q1_var[105]
   + total_stock_second_period_105 = 0
 R1057: total_stock_second_period_105 + Lost_1_expr_105
   = 422.4521598086443
 R1058: Sold_1[105] + Left_1[105] - total_stock_second_period_105 = 0
 R1059: - 700 Sold_0[105] - 700 Sold_1[105] + 700 Lost_0[105]
   + 700 Lost_1[105] + 300 Left_1[105] + profits_vars[105] = 0
 Fraction_lower_bound_106: Fraction[106] >= 0
 Fraction_upper_bound_106: Fraction[106] <= 1
 R1062: 0.6858581268627552 alphas[0] - 0.4872098249551625 alphas[1]
   + 0.7696150229027544 alphas[2] + 0.6254475630316598 alphas[3]
   - alphas[4] + f_var[106] = 0
 R1063: - 2094.959652373807 Fraction[106] + Q0_var[106] = 0
 R1064: Q0_var[106] + Lost_0_expr_106 = 1129.009875204798
 R1065: Q0_var[106] + Q_hat_adjusted_106 = 1359.679560406192
 R1066: Sold_0[106] - Q0_var[106] - Q1_var[106]
   + total_stock_second_period_106 = 0
 R1067: total_stock_second_period_106 + Lost_1_expr_106
   = 230.6590717572131
 R1068: Sold_1[106] + Left_1[106] - total_stock_second_period_106 = 0
 R1069: - 700 Sold_0[106] - 700 Sold_1[106] + 700 Lost_0[106]
   + 700 Lost_1[106] + 300 Left_1[106] + profits_vars[106] = 0
 Fraction_lower_bound_107: Fraction[107] >= 0
 Fraction_upper_bound_107: Fraction[107] <= 1
 R1072: 0.8412142229242753 alphas[0] - 0.2957128561764039 alphas[1]
   - 0.5343651210641944 alphas[2] + 0.2547157308392952 alphas[3]
   - alphas[4] + f_var[107] = 0
 R1073: - 2094.959652373807 Fraction[107] + Q0_var[107] = 0
 R1074: Q0_var[107] + Lost_0_expr_107 = 1489.323294919429
 R1075: Q0_var[107] + Q_hat_adjusted_107 = 1905.414643802464
 R1076: Sold_0[107] - Q0_var[107] - Q1_var[107]
   + total_stock_second_period_107 = 0
 R1077: total_stock_second_period_107 + Lost_1_expr_107
   = 416.0807377570629
 R1078: Sold_1[107] + Left_1[107] - total_stock_second_period_107 = 0
 R1079: - 700 Sold_0[107] - 700 Sold_1[107] + 700 Lost_0[107]
   + 700 Lost_1[107] + 300 Left_1[107] + profits_vars[107] = 0
 Fraction_lower_bound_108: Fraction[108] >= 0
 Fraction_upper_bound_108: Fraction[108] <= 1
 R1082: - 0.0997123744235002 alphas[0] - 0.3240413151880734 alphas[1]
   - 1.514414209781231 alphas[2] + 1.098763783252958 alphas[3] - alphas[4]
   + f_var[108] = 0
 R1083: - 2094.959652373807 Fraction[108] + Q0_var[108] = 0
 R1084: Q0_var[108] + Lost_0_expr_108 = 2423.338290859486
 R1085: Q0_var[108] + Q_hat_adjusted_108 = 2978.789379187531
 R1086: Sold_0[108] - Q0_var[108] - Q1_var[108]
   + total_stock_second_period_108 = 0
 R1087: total_stock_second_period_108 + Lost_1_expr_108
   = 555.4404784956855
 R1088: Sold_1[108] + Left_1[108] - total_stock_second_period_108 = 0
 R1089: - 700 Sold_0[108] - 700 Sold_1[108] + 700 Lost_0[108]
   + 700 Lost_1[108] + 300 Left_1[108] + profits_vars[108] = 0
 Fraction_lower_bound_109: Fraction[109] >= 0
 Fraction_upper_bound_109: Fraction[109] <= 1
 R1092: 0.3998390453176563 alphas[0] - 0.9626559482830178 alphas[1]
   - 0.4746040776779045 alphas[2] + 1.192850380097138 alphas[3] - alphas[4]
   + f_var[109] = 0
 R1093: - 2094.959652373807 Fraction[109] + Q0_var[109] = 0
 R1094: Q0_var[109] + Lost_0_expr_109 = 1347.512484001371
 R1095: Q0_var[109] + Q_hat_adjusted_109 = 1755.106010468269
 R1096: Sold_0[109] - Q0_var[109] - Q1_var[109]
   + total_stock_second_period_109 = 0
 R1097: total_stock_second_period_109 + Lost_1_expr_109
   = 407.5829149352639
 R1098: Sold_1[109] + Left_1[109] - total_stock_second_period_109 = 0
 R1099: - 700 Sold_0[109] - 700 Sold_1[109] + 700 Lost_0[109]
   + 700 Lost_1[109] + 300 Left_1[109] + profits_vars[109] = 0
 Fraction_lower_bound_110: Fraction[110] >= 0
 Fraction_upper_bound_110: Fraction[110] <= 1
 R1102: 0.0979229866434775 alphas[0] - 1.442371279985808 alphas[1]
   + 1.041334184526679 alphas[2] + 0.8793101653453025 alphas[3] - alphas[4]
   + f_var[110] = 0
 R1103: - 2094.959652373807 Fraction[110] + Q0_var[110] = 0
 R1104: Q0_var[110] + Lost_0_expr_110 = 808.6962681387486
 R1105: Q0_var[110] + Q_hat_adjusted_110 = 1000.728387425502
 R1106: Sold_0[110] - Q0_var[110] - Q1_var[110]
   + total_stock_second_period_110 = 0
 R1107: total_stock_second_period_110 + Lost_1_expr_110
   = 192.0215051734302
 R1108: Sold_1[110] + Left_1[110] - total_stock_second_period_110 = 0
 R1109: - 700 Sold_0[110] - 700 Sold_1[110] + 700 Lost_0[110]
   + 700 Lost_1[110] + 300 Left_1[110] + profits_vars[110] = 0
 Fraction_lower_bound_111: Fraction[111] >= 0
 Fraction_upper_bound_111: Fraction[111] <= 1
 R1112: 0.5036110761409982 alphas[0] - 0.8130312742513084 alphas[1]
   + 0.611931083025184 alphas[2] - 0.1197376889329673 alphas[3] - alphas[4]
   + f_var[111] = 0
 R1113: - 2094.959652373807 Fraction[111] + Q0_var[111] = 0
 R1114: Q0_var[111] + Lost_0_expr_111 = 1081.933542856184
 R1115: Q0_var[111] + Q_hat_adjusted_111 = 1335.025364569353
 R1116: Sold_0[111] - Q0_var[111] - Q1_var[111]
   + total_stock_second_period_111 = 0
 R1117: total_stock_second_period_111 + Lost_1_expr_111
   = 253.0812067634787
 R1118: Sold_1[111] + Left_1[111] - total_stock_second_period_111 = 0
 R1119: - 700 Sold_0[111] - 700 Sold_1[111] + 700 Lost_0[111]
   + 700 Lost_1[111] + 300 Left_1[111] + profits_vars[111] = 0
 Fraction_lower_bound_112: Fraction[112] >= 0
 Fraction_upper_bound_112: Fraction[112] <= 1
 R1122: - 1.653661811080881 alphas[0] + 1.10098270475632 alphas[1]
   + 0.4523070478606979 alphas[2] - 0.9689966237605661 alphas[3]
   - alphas[4] + f_var[112] = 0
 R1123: - 2094.959652373807 Fraction[112] + Q0_var[112] = 0
 R1124: Q0_var[112] + Lost_0_expr_112 = 1086.471625591272
 R1125: Q0_var[112] + Q_hat_adjusted_112 = 1362.261455648779
 R1126: Sold_0[112] - Q0_var[112] - Q1_var[112]
   + total_stock_second_period_112 = 0
 R1127: total_stock_second_period_112 + Lost_1_expr_112
   = 275.7792169063174
 R1128: Sold_1[112] + Left_1[112] - total_stock_second_period_112 = 0
 R1129: - 700 Sold_0[112] - 700 Sold_1[112] + 700 Lost_0[112]
   + 700 Lost_1[112] + 300 Left_1[112] + profits_vars[112] = 0
 Fraction_lower_bound_113: Fraction[113] >= 0
 Fraction_upper_bound_113: Fraction[113] <= 1
 R1132: 1.198957356027284 alphas[0] - 1.498118617643345 alphas[1]
   + 0.5943254173896947 alphas[2] + 0.754274341589967 alphas[3] - alphas[4]
   + f_var[113] = 0
 R1133: - 2094.959652373807 Fraction[113] + Q0_var[113] = 0
 R1134: Q0_var[113] + Lost_0_expr_113 = 920.7942813195045
 R1135: Q0_var[113] + Q_hat_adjusted_113 = 1176.389569889023
 R1136: Sold_0[113] - Q0_var[113] - Q1_var[113]
   + total_stock_second_period_113 = 0
 R1137: total_stock_second_period_113 + Lost_1_expr_113
   = 255.5846742191091
 R1138: Sold_1[113] + Left_1[113] - total_stock_second_period_113 = 0
 R1139: - 700 Sold_0[113] - 700 Sold_1[113] + 700 Lost_0[113]
   + 700 Lost_1[113] + 300 Left_1[113] + profits_vars[113] = 0
 Fraction_lower_bound_114: Fraction[114] >= 0
 Fraction_upper_bound_114: Fraction[114] <= 1
 R1142: 1.323808364044422 alphas[0] - 1.201223195195274 alphas[1]
   - 0.7180754611156209 alphas[2] - 1.053134951841229 alphas[3] - alphas[4]
   + f_var[114] = 0
 R1143: - 2094.959652373807 Fraction[114] + Q0_var[114] = 0
 R1144: Q0_var[114] + Lost_0_expr_114 = 1705.059422043449
 R1145: Q0_var[114] + Q_hat_adjusted_114 = 2147.273776456608
 R1146: Sold_0[114] - Q0_var[114] - Q1_var[114]
   + total_stock_second_period_114 = 0
 R1147: total_stock_second_period_114 + Lost_1_expr_114
   = 442.2037408344759
 R1148: Sold_1[114] + Left_1[114] - total_stock_second_period_114 = 0
 R1149: - 700 Sold_0[114] - 700 Sold_1[114] + 700 Lost_0[114]
   + 700 Lost_1[114] + 300 Left_1[114] + profits_vars[114] = 0
 Fraction_lower_bound_115: Fraction[115] >= 0
 Fraction_upper_bound_115: Fraction[115] <= 1
 R1152: 0.4834554178304724 alphas[0] + 0.6760100572817818 alphas[1]
   - 1.651976561537687 alphas[2] + 1.498739775277316 alphas[3] - alphas[4]
   + f_var[115] = 0
 R1153: - 2094.959652373807 Fraction[115] + Q0_var[115] = 0
 R1154: Q0_var[115] + Lost_0_expr_115 = 2231.806415904008
 R1155: Q0_var[115] + Q_hat_adjusted_115 = 2806.818415228732
 R1156: Sold_0[115] - Q0_var[115] - Q1_var[115]
   + total_stock_second_period_115 = 0
 R1157: total_stock_second_period_115 + Lost_1_expr_115
   = 575.0013901855997
 R1158: Sold_1[115] + Left_1[115] - total_stock_second_period_115 = 0
 R1159: - 700 Sold_0[115] - 700 Sold_1[115] + 700 Lost_0[115]
   + 700 Lost_1[115] + 300 Left_1[115] + profits_vars[115] = 0
 Fraction_lower_bound_116: Fraction[116] >= 0
 Fraction_upper_bound_116: Fraction[116] <= 1
 R1162: - 0.3604669782669481 alphas[0] + 1.264864500267102 alphas[1]
   - 0.0834006568511118 alphas[2] + 0.9078218145181578 alphas[3]
   - alphas[4] + f_var[116] = 0
 R1163: - 2094.959652373807 Fraction[116] + Q0_var[116] = 0
 R1164: Q0_var[116] + Lost_0_expr_116 = 1223.444593705643
 R1165: Q0_var[116] + Q_hat_adjusted_116 = 1575.410287565999
 R1166: Sold_0[116] - Q0_var[116] - Q1_var[116]
   + total_stock_second_period_116 = 0
 R1167: total_stock_second_period_116 + Lost_1_expr_116
   = 351.9550816663686
 R1168: Sold_1[116] + Left_1[116] - total_stock_second_period_116 = 0
 R1169: - 700 Sold_0[116] - 700 Sold_1[116] + 700 Lost_0[116]
   + 700 Lost_1[116] + 300 Left_1[116] + profits_vars[116] = 0
 Fraction_lower_bound_117: Fraction[117] >= 0
 Fraction_upper_bound_117: Fraction[117] <= 1
 R1172: - 0.5927156881375353 alphas[0] - 1.004669317245455 alphas[1]
   - 1.090269663547613 alphas[2] + 0.2436570156402071 alphas[3] - alphas[4]
   + f_var[117] = 0
 R1173: - 2094.959652373807 Fraction[117] + Q0_var[117] = 0
 R1174: Q0_var[117] + Lost_0_expr_117 = 2015.064072540993
 R1175: Q0_var[117] + Q_hat_adjusted_117 = 2510.203209556999
 R1176: Sold_0[117] - Q0_var[117] - Q1_var[117]
   + total_stock_second_period_117 = 0
 R1177: total_stock_second_period_117 + Lost_1_expr_117
   = 495.1285260850203
 R1178: Sold_1[117] + Left_1[117] - total_stock_second_period_117 = 0
 R1179: - 700 Sold_0[117] - 700 Sold_1[117] + 700 Lost_0[117]
   + 700 Lost_1[117] + 300 Left_1[117] + profits_vars[117] = 0
 Fraction_lower_bound_118: Fraction[118] >= 0
 Fraction_upper_bound_118: Fraction[118] <= 1
 R1182: 0.2998874093246034 alphas[0] - 1.570149669337666 alphas[1]
   - 1.632223262094059 alphas[2] + 0.5210900304131025 alphas[3] - alphas[4]
   + f_var[118] = 0
 R1183: - 2094.959652373807 Fraction[118] + Q0_var[118] = 0
 R1184: Q0_var[118] + Lost_0_expr_118 = 2209.933624128001
 R1185: Q0_var[118] + Q_hat_adjusted_118 = 2782.136769930057
 R1186: Sold_0[118] - Q0_var[118] - Q1_var[118]
   + total_stock_second_period_118 = 0
 R1187: total_stock_second_period_118 + Lost_1_expr_118 = 572.19253630176
 R1188: Sold_1[118] + Left_1[118] - total_stock_second_period_118 = 0
 R1189: - 700 Sold_0[118] - 700 Sold_1[118] + 700 Lost_0[118]
   + 700 Lost_1[118] + 300 Left_1[118] + profits_vars[118] = 0
 Fraction_lower_bound_119: Fraction[119] >= 0
 Fraction_upper_bound_119: Fraction[119] <= 1
 R1192: - 1.751756582907006 alphas[0] + 1.247488704723383 alphas[1]
   + 0.3718340740476324 alphas[2] + 0.7562302403101655 alphas[3]
   - alphas[4] + f_var[119] = 0
 R1193: - 2094.959652373807 Fraction[119] + Q0_var[119] = 0
 R1194: Q0_var[119] + Lost_0_expr_119 = 1026.285556894711
 R1195: Q0_var[119] + Q_hat_adjusted_119 = 1313.518376244418
 R1196: Sold_0[119] - Q0_var[119] - Q1_var[119]
   + total_stock_second_period_119 = 0
 R1197: total_stock_second_period_119 + Lost_1_expr_119 = 287.222207768883
 R1198: Sold_1[119] + Left_1[119] - total_stock_second_period_119 = 0
 R1199: - 700 Sold_0[119] - 700 Sold_1[119] + 700 Lost_0[119]
   + 700 Lost_1[119] + 300 Left_1[119] + profits_vars[119] = 0
 Fraction_lower_bound_120: Fraction[120] >= 0
 Fraction_upper_bound_120: Fraction[120] <= 1
 R1202: 0.4544837257415876 alphas[0] - 0.4914014835794792 alphas[1]
   + 1.525748794897156 alphas[2] + 0.4230661740835008 alphas[3] - alphas[4]
   + f_var[120] = 0
 R1203: - 2094.959652373807 Fraction[120] + Q0_var[120] = 0
 R1204: Q0_var[120] + Lost_0_expr_120 = 312.9511939622993
 R1205: Q0_var[120] + Q_hat_adjusted_120 = 436.1011569248428
 R1206: Sold_0[120] - Q0_var[120] - Q1_var[120]
   + total_stock_second_period_120 = 0
 R1207: total_stock_second_period_120 + Lost_1_expr_120
   = 123.1393488977077
 R1208: Sold_1[120] + Left_1[120] - total_stock_second_period_120 = 0
 R1209: - 700 Sold_0[120] - 700 Sold_1[120] + 700 Lost_0[120]
   + 700 Lost_1[120] + 300 Left_1[120] + profits_vars[120] = 0
 Fraction_lower_bound_121: Fraction[121] >= 0
 Fraction_upper_bound_121: Fraction[121] <= 1
 R1212: - 0.8047916051798193 alphas[0] + 1.638005318151286 alphas[1]
   - 0.9603248440714964 alphas[2] - 1.182890942039465 alphas[3] - alphas[4]
   + f_var[121] = 0
 R1213: - 2094.959652373807 Fraction[121] + Q0_var[121] = 0
 R1214: Q0_var[121] + Lost_0_expr_121 = 1965.264986901952
 R1215: Q0_var[121] + Q_hat_adjusted_121 = 2441.926400295046
 R1216: Sold_0[121] - Q0_var[121] - Q1_var[121]
   + total_stock_second_period_121 = 0
 R1217: total_stock_second_period_121 + Lost_1_expr_121
   = 476.6508023046482
 R1218: Sold_1[121] + Left_1[121] - total_stock_second_period_121 = 0
 R1219: - 700 Sold_0[121] - 700 Sold_1[121] + 700 Lost_0[121]
   + 700 Lost_1[121] + 300 Left_1[121] + profits_vars[121] = 0
 Fraction_lower_bound_122: Fraction[122] >= 0
 Fraction_upper_bound_122: Fraction[122] <= 1
 R1222: - 0.5191256882073663 alphas[0] + 1.552845240528113 alphas[1]
   - 0.8959377071970059 alphas[2] + 1.085943637207431 alphas[3] - alphas[4]
   + f_var[122] = 0
 R1223: - 2094.959652373807 Fraction[122] + Q0_var[122] = 0
 R1224: Q0_var[122] + Lost_0_expr_122 = 2079.300835290032
 R1225: Q0_var[122] + Q_hat_adjusted_122 = 2546.806612398417
 R1226: Sold_0[122] - Q0_var[122] - Q1_var[122]
   + total_stock_second_period_122 = 0
 R1227: total_stock_second_period_122 + Lost_1_expr_122
   = 467.4951642809676
 R1228: Sold_1[122] + Left_1[122] - total_stock_second_period_122 = 0
 R1229: - 700 Sold_0[122] - 700 Sold_1[122] + 700 Lost_0[122]
   + 700 Lost_1[122] + 300 Left_1[122] + profits_vars[122] = 0
 Fraction_lower_bound_123: Fraction[123] >= 0
 Fraction_upper_bound_123: Fraction[123] <= 1
 R1232: - 1.117118485053687 alphas[0] - 0.9355999803669214 alphas[1]
   + 0.6221663273045357 alphas[2] + 0.1878933568516148 alphas[3]
   - alphas[4] + f_var[123] = 0
 R1233: - 2094.959652373807 Fraction[123] + Q0_var[123] = 0
 R1234: Q0_var[123] + Lost_0_expr_123 = 1074.558854238668
 R1235: Q0_var[123] + Q_hat_adjusted_123 = 1326.195255685581
 R1236: Sold_0[123] - Q0_var[123] - Q1_var[123]
   + total_stock_second_period_123 = 0
 R1237: total_stock_second_period_123 + Lost_1_expr_123
   = 251.6257888608569
 R1238: Sold_1[123] + Left_1[123] - total_stock_second_period_123 = 0
 R1239: - 700 Sold_0[123] - 700 Sold_1[123] + 700 Lost_0[123]
   + 700 Lost_1[123] + 300 Left_1[123] + profits_vars[123] = 0
 Fraction_lower_bound_124: Fraction[124] >= 0
 Fraction_upper_bound_124: Fraction[124] <= 1
 R1242: - 1.673195941455682 alphas[0] - 1.03651132006734 alphas[1]
   + 0.7421469430390375 alphas[2] - 0.5497734388509558 alphas[3]
   - alphas[4] + f_var[124] = 0
 R1243: - 2094.959652373807 Fraction[124] + Q0_var[124] = 0
 R1244: Q0_var[124] + Lost_0_expr_124 = 735.7964543736533
 R1245: Q0_var[124] + Q_hat_adjusted_124 = 970.3720088897774
 R1246: Sold_0[124] - Q0_var[124] - Q1_var[124]
   + total_stock_second_period_124 = 0
 R1247: total_stock_second_period_124 + Lost_1_expr_124
   = 234.5649419146513
 R1248: Sold_1[124] + Left_1[124] - total_stock_second_period_124 = 0
 R1249: - 700 Sold_0[124] - 700 Sold_1[124] + 700 Lost_0[124]
   + 700 Lost_1[124] + 300 Left_1[124] + profits_vars[124] = 0
 Fraction_lower_bound_125: Fraction[125] >= 0
 Fraction_upper_bound_125: Fraction[125] <= 1
 R1252: - 1.378642038464481 alphas[0] + 0.5110714935219823 alphas[1]
   - 0.7846054028473677 alphas[2] + 0.2717467237749861 alphas[3]
   - alphas[4] + f_var[125] = 0
 R1253: - 2094.959652373807 Fraction[125] + Q0_var[125] = 0
 R1254: Q0_var[125] + Lost_0_expr_125 = 1458.777435761754
 R1255: Q0_var[125] + Q_hat_adjusted_125 = 1910.452125106315
 R1256: Sold_0[125] - Q0_var[125] - Q1_var[125]
   + total_stock_second_period_125 = 0
 R1257: total_stock_second_period_125 + Lost_1_expr_125
   = 451.6640786255443
 R1258: Sold_1[125] + Left_1[125] - total_stock_second_period_125 = 0
 R1259: - 700 Sold_0[125] - 700 Sold_1[125] + 700 Lost_0[125]
   + 700 Lost_1[125] + 300 Left_1[125] + profits_vars[125] = 0
 Fraction_lower_bound_126: Fraction[126] >= 0
 Fraction_upper_bound_126: Fraction[126] <= 1
 R1262: - 0.9518617054150537 alphas[0] + 0.1837411552608302 alphas[1]
   + 1.138516043325392 alphas[2] - 1.460547982889029 alphas[3] - alphas[4]
   + f_var[126] = 0
 R1263: - 2094.959652373807 Fraction[126] + Q0_var[126] = 0
 R1264: Q0_var[126] + Lost_0_expr_126 = 592.1874092484778
 R1265: Q0_var[126] + Q_hat_adjusted_126 = 770.4005895344142
 R1266: Sold_0[126] - Q0_var[126] - Q1_var[126]
   + total_stock_second_period_126 = 0
 R1267: total_stock_second_period_126 + Lost_1_expr_126
   = 178.2025661024041
 R1268: Sold_1[126] + Left_1[126] - total_stock_second_period_126 = 0
 R1269: - 700 Sold_0[126] - 700 Sold_1[126] + 700 Lost_0[126]
   + 700 Lost_1[126] + 300 Left_1[126] + profits_vars[126] = 0
 Fraction_lower_bound_127: Fraction[127] >= 0
 Fraction_upper_bound_127: Fraction[127] <= 1
 R1272: - 0.7258702017721038 alphas[0] - 0.916663329583586 alphas[1]
   - 1.277303452501748 alphas[2] + 1.384250612389809 alphas[3] - alphas[4]
   + f_var[127] = 0
 R1273: - 2094.959652373807 Fraction[127] + Q0_var[127] = 0
 R1274: Q0_var[127] + Lost_0_expr_127 = 2051.28196134345
 R1275: Q0_var[127] + Q_hat_adjusted_127 = 2573.016682414295
 R1276: Sold_0[127] - Q0_var[127] - Q1_var[127]
   + total_stock_second_period_127 = 0
 R1277: total_stock_second_period_127 + Lost_1_expr_127
   = 521.7241126103153
 R1278: Sold_1[127] + Left_1[127] - total_stock_second_period_127 = 0
 R1279: - 700 Sold_0[127] - 700 Sold_1[127] + 700 Lost_0[127]
   + 700 Lost_1[127] + 300 Left_1[127] + profits_vars[127] = 0
 Fraction_lower_bound_128: Fraction[128] >= 0
 Fraction_upper_bound_128: Fraction[128] <= 1
 R1282: 0.5103560849131781 alphas[0] - 1.17632387879674 alphas[1]
   - 1.337715558248699 alphas[2] + 0.2858377775757389 alphas[3] - alphas[4]
   + f_var[128] = 0
 R1283: - 2094.959652373807 Fraction[128] + Q0_var[128] = 0
 R1284: Q0_var[128] + Lost_0_expr_128 = 2263.234217520333
 R1285: Q0_var[128] + Q_hat_adjusted_128 = 2793.559341236527
 R1286: Sold_0[128] - Q0_var[128] - Q1_var[128]
   + total_stock_second_period_128 = 0
 R1287: total_stock_second_period_128 + Lost_1_expr_128
   = 530.3145143476308
 R1288: Sold_1[128] + Left_1[128] - total_stock_second_period_128 = 0
 R1289: - 700 Sold_0[128] - 700 Sold_1[128] + 700 Lost_0[128]
   + 700 Lost_1[128] + 300 Left_1[128] + profits_vars[128] = 0
 Fraction_lower_bound_129: Fraction[129] >= 0
 Fraction_upper_bound_129: Fraction[129] <= 1
 R1292: 1.150407539958565 alphas[0] + 0.7541952007160704 alphas[1]
   - 0.3029495915446996 alphas[2] - 1.644061295770889 alphas[3] - alphas[4]
   + f_var[129] = 0
 R1293: - 2094.959652373807 Fraction[129] + Q0_var[129] = 0
 R1294: Q0_var[129] + Lost_0_expr_129 = 1837.421329608244
 R1295: Q0_var[129] + Q_hat_adjusted_129 = 2220.606154339438
 R1296: Sold_0[129] - Q0_var[129] - Q1_var[129]
   + total_stock_second_period_129 = 0
 R1297: total_stock_second_period_129 + Lost_1_expr_129
   = 383.1742144328632
 R1298: Sold_1[129] + Left_1[129] - total_stock_second_period_129 = 0
 R1299: - 700 Sold_0[129] - 700 Sold_1[129] + 700 Lost_0[129]
   + 700 Lost_1[129] + 300 Left_1[129] + profits_vars[129] = 0
 Fraction_lower_bound_130: Fraction[130] >= 0
 Fraction_upper_bound_130: Fraction[130] <= 1
 R1302: 1.440250217161031 alphas[0] - 0.9326314010464299 alphas[1]
   + 0.9191021780173747 alphas[2] + 0.087730665934129 alphas[3] - alphas[4]
   + f_var[130] = 0
 R1303: - 2094.959652373807 Fraction[130] + Q0_var[130] = 0
 R1304: Q0_var[130] + Lost_0_expr_130 = 1141.787276912816
 R1305: Q0_var[130] + Q_hat_adjusted_130 = 1351.200382094416
 R1306: Sold_0[130] - Q0_var[130] - Q1_var[130]
   + total_stock_second_period_130 = 0
 R1307: total_stock_second_period_130 + Lost_1_expr_130 = 209.40249244818
 R1308: Sold_1[130] + Left_1[130] - total_stock_second_period_130 = 0
 R1309: - 700 Sold_0[130] - 700 Sold_1[130] + 700 Lost_0[130]
   + 700 Lost_1[130] + 300 Left_1[130] + profits_vars[130] = 0
 Fraction_lower_bound_131: Fraction[131] >= 0
 Fraction_upper_bound_131: Fraction[131] <= 1
 R1312: 0.8293259337682342 alphas[0] + 1.00388199587397 alphas[1]
   - 1.64846127464022 alphas[2] - 1.215677673220806 alphas[3] - alphas[4]
   + f_var[131] = 0
 R1313: - 2094.959652373807 Fraction[131] + Q0_var[131] = 0
 R1314: Q0_var[131] + Lost_0_expr_131 = 2318.382377519079
 R1315: Q0_var[131] + Q_hat_adjusted_131 = 2892.894514522092
 R1316: Sold_0[131] - Q0_var[131] - Q1_var[131]
   + total_stock_second_period_131 = 0
 R1317: total_stock_second_period_131 + Lost_1_expr_131 = 574.501528009008
 R1318: Sold_1[131] + Left_1[131] - total_stock_second_period_131 = 0
 R1319: - 700 Sold_0[131] - 700 Sold_1[131] + 700 Lost_0[131]
   + 700 Lost_1[131] + 300 Left_1[131] + profits_vars[131] = 0
 Fraction_lower_bound_132: Fraction[132] >= 0
 Fraction_upper_bound_132: Fraction[132] <= 1
 R1322: 0.1805256999078004 alphas[0] + 0.1857928932784812 alphas[1]
   + 0.8611662544426599 alphas[2] + 1.494404784117802 alphas[3] - alphas[4]
   + f_var[132] = 0
 R1323: - 2094.959652373807 Fraction[132] + Q0_var[132] = 0
 R1324: Q0_var[132] + Lost_0_expr_132 = 966.2327486166023
 R1325: Q0_var[132] + Q_hat_adjusted_132 = 1183.884151170225
 R1326: Sold_0[132] - Q0_var[132] - Q1_var[132]
   + total_stock_second_period_132 = 0
 R1327: total_stock_second_period_132 + Lost_1_expr_132
   = 217.6407892664828
 R1328: Sold_1[132] + Left_1[132] - total_stock_second_period_132 = 0
 R1329: - 700 Sold_0[132] - 700 Sold_1[132] + 700 Lost_0[132]
   + 700 Lost_1[132] + 300 Left_1[132] + profits_vars[132] = 0
 Fraction_lower_bound_133: Fraction[133] >= 0
 Fraction_upper_bound_133: Fraction[133] <= 1
 R1332: - 0.1252156936626839 alphas[0] - 0.7155801834530185 alphas[1]
   + 0.6019497439984899 alphas[2] - 0.8546817587489093 alphas[3]
   - alphas[4] + f_var[133] = 0
 R1333: - 2094.959652373807 Fraction[133] + Q0_var[133] = 0
 R1334: Q0_var[133] + Lost_0_expr_133 = 973.7608047145944
 R1335: Q0_var[133] + Q_hat_adjusted_133 = 1228.271938513793
 R1336: Sold_0[133] - Q0_var[133] - Q1_var[133]
   + total_stock_second_period_133 = 0
 R1337: total_stock_second_period_133 + Lost_1_expr_133
   = 254.5005201784956
 R1338: Sold_1[133] + Left_1[133] - total_stock_second_period_133 = 0
 R1339: - 700 Sold_0[133] - 700 Sold_1[133] + 700 Lost_0[133]
   + 700 Lost_1[133] + 300 Left_1[133] + profits_vars[133] = 0
 Fraction_lower_bound_134: Fraction[134] >= 0
 Fraction_upper_bound_134: Fraction[134] <= 1
 R1342: - 0.9809026203067966 alphas[0] - 0.6057151817332193 alphas[1]
   + 0.1348818925239168 alphas[2] + 0.7519007625189389 alphas[3]
   - alphas[4] + f_var[134] = 0
 R1343: - 2094.959652373807 Fraction[134] + Q0_var[134] = 0
 R1344: Q0_var[134] + Lost_0_expr_134 = 1376.996051280214
 R1345: Q0_var[134] + Q_hat_adjusted_134 = 1697.922688497724
 R1346: Sold_0[134] - Q0_var[134] - Q1_var[134]
   + total_stock_second_period_134 = 0
 R1347: total_stock_second_period_134 + Lost_1_expr_134
   = 320.9160247059754
 R1348: Sold_1[134] + Left_1[134] - total_stock_second_period_134 = 0
 R1349: - 700 Sold_0[134] - 700 Sold_1[134] + 700 Lost_0[134]
   + 700 Lost_1[134] + 300 Left_1[134] + profits_vars[134] = 0
 Fraction_lower_bound_135: Fraction[135] >= 0
 Fraction_upper_bound_135: Fraction[135] <= 1
 R1352: - 1.613609236193847 alphas[0] + 0.00340783784402844 alphas[1]
   - 0.142543329548259 alphas[2] + 1.453823716686668 alphas[3] - alphas[4]
   + f_var[135] = 0
 R1353: - 2094.959652373807 Fraction[135] + Q0_var[135] = 0
 R1354: Q0_var[135] + Lost_0_expr_135 = 1302.516793859506
 R1355: Q0_var[135] + Q_hat_adjusted_135 = 1662.892379079373
 R1356: Sold_0[135] - Q0_var[135] - Q1_var[135]
   + total_stock_second_period_135 = 0
 R1357: total_stock_second_period_135 + Lost_1_expr_135
   = 360.3649742202745
 R1358: Sold_1[135] + Left_1[135] - total_stock_second_period_135 = 0
 R1359: - 700 Sold_0[135] - 700 Sold_1[135] + 700 Lost_0[135]
   + 700 Lost_1[135] + 300 Left_1[135] + profits_vars[135] = 0
 Fraction_lower_bound_136: Fraction[136] >= 0
 Fraction_upper_bound_136: Fraction[136] <= 1
 R1362: 1.253889553019911 alphas[0] - 0.5528552300804705 alphas[1]
   + 0.0867609682697506 alphas[2] - 0.5004664726691301 alphas[3]
   - alphas[4] + f_var[136] = 0
 R1363: - 2094.959652373807 Fraction[136] + Q0_var[136] = 0
 R1364: Q0_var[136] + Lost_0_expr_136 = 1505.699408066973
 R1365: Q0_var[136] + Q_hat_adjusted_136 = 1833.468680946892
 R1366: Sold_0[136] - Q0_var[136] - Q1_var[136]
   + total_stock_second_period_136 = 0
 R1367: total_stock_second_period_136 + Lost_1_expr_136 = 327.758661065081
 R1368: Sold_1[136] + Left_1[136] - total_stock_second_period_136 = 0
 R1369: - 700 Sold_0[136] - 700 Sold_1[136] + 700 Lost_0[136]
   + 700 Lost_1[136] + 300 Left_1[136] + profits_vars[136] = 0
 Fraction_lower_bound_137: Fraction[137] >= 0
 Fraction_upper_bound_137: Fraction[137] <= 1
 R1372: 1.289046879258543 alphas[0] + 1.502922994148656 alphas[1]
   + 1.687748866252984 alphas[2] - 1.110757593886217 alphas[3] - alphas[4]
   + f_var[137] = 0
 R1373: - 2094.959652373807 Fraction[137] + Q0_var[137] = 0
 R1374: Q0_var[137] + Lost_0_expr_137 = 538.5244300286142
 R1375: Q0_var[137] + Q_hat_adjusted_137 = 638.6385204732308
 R1376: Sold_0[137] - Q0_var[137] - Q1_var[137]
   + total_stock_second_period_137 = 0
 R1377: total_stock_second_period_137 + Lost_1_expr_137
   = 100.1034742410803
 R1378: Sold_1[137] + Left_1[137] - total_stock_second_period_137 = 0
 R1379: - 700 Sold_0[137] - 700 Sold_1[137] + 700 Lost_0[137]
   + 700 Lost_1[137] + 300 Left_1[137] + profits_vars[137] = 0
 Fraction_lower_bound_138: Fraction[138] >= 0
 Fraction_upper_bound_138: Fraction[138] <= 1
 R1382: - 0.3559270144176451 alphas[0] + 0.7138108319901729 alphas[1]
   - 1.343227047126376 alphas[2] + 1.271447315116746 alphas[3] - alphas[4]
   + f_var[138] = 0
 R1383: - 2094.959652373807 Fraction[138] + Q0_var[138] = 0
 R1384: Q0_var[138] + Lost_0_expr_138 = 2357.305547418276
 R1385: Q0_var[138] + Q_hat_adjusted_138 = 2888.414385975916
 R1386: Sold_0[138] - Q0_var[138] - Q1_var[138]
   + total_stock_second_period_138 = 0
 R1387: total_stock_second_period_138 + Lost_1_expr_138
   = 531.0982298471089
 R1388: Sold_1[138] + Left_1[138] - total_stock_second_period_138 = 0
 R1389: - 700 Sold_0[138] - 700 Sold_1[138] + 700 Lost_0[138]
   + 700 Lost_1[138] + 300 Left_1[138] + profits_vars[138] = 0
 Fraction_lower_bound_139: Fraction[139] >= 0
 Fraction_upper_bound_139: Fraction[139] <= 1
 R1392: - 0.8865525467155236 alphas[0] + 0.9527816209359854 alphas[1]
   + 1.605005476785476 alphas[2] + 1.503081794648911 alphas[3] - alphas[4]
   + f_var[139] = 0
 R1393: - 2094.959652373807 Fraction[139] + Q0_var[139] = 0
 R1394: Q0_var[139] + Lost_0_expr_139 = 645.2420193336925
 R1395: Q0_var[139] + Q_hat_adjusted_139 = 757.1219436758093
 R1396: Sold_0[139] - Q0_var[139] - Q1_var[139]
   + total_stock_second_period_139 = 0
 R1397: total_stock_second_period_139 + Lost_1_expr_139
   = 111.8693107045134
 R1398: Sold_1[139] + Left_1[139] - total_stock_second_period_139 = 0
 R1399: - 700 Sold_0[139] - 700 Sold_1[139] + 700 Lost_0[139]
   + 700 Lost_1[139] + 300 Left_1[139] + profits_vars[139] = 0
 Fraction_lower_bound_140: Fraction[140] >= 0
 Fraction_upper_bound_140: Fraction[140] <= 1
 R1402: - 1.236724825932385 alphas[0] + 1.399160998677426 alphas[1]
   - 1.411281196674741 alphas[2] - 1.553194700683223 alphas[3] - alphas[4]
   + f_var[140] = 0
 R1403: - 2094.959652373807 Fraction[140] + Q0_var[140] = 0
 R1404: Q0_var[140] + Lost_0_expr_140 = 2100.821620038143
 R1405: Q0_var[140] + Q_hat_adjusted_140 = 2641.607534948665
 R1406: Sold_0[140] - Q0_var[140] - Q1_var[140]
   + total_stock_second_period_140 = 0
 R1407: total_stock_second_period_140 + Lost_1_expr_140
   = 540.7753049512376
 R1408: Sold_1[140] + Left_1[140] - total_stock_second_period_140 = 0
 R1409: - 700 Sold_0[140] - 700 Sold_1[140] + 700 Lost_0[140]
   + 700 Lost_1[140] + 300 Left_1[140] + profits_vars[140] = 0
 Fraction_lower_bound_141: Fraction[141] >= 0
 Fraction_upper_bound_141: Fraction[141] <= 1
 R1412: - 1.535537589994194 alphas[0] + 0.9812739530390894 alphas[1]
   - 1.204761399866132 alphas[2] + 0.3153816266657428 alphas[3] - alphas[4]
   + f_var[141] = 0
 R1413: - 2094.959652373807 Fraction[141] + Q0_var[141] = 0
 R1414: Q0_var[141] + Lost_0_expr_141 = 2414.018727910669
 R1415: Q0_var[141] + Q_hat_adjusted_141 = 2925.438209452487
 R1416: Sold_0[141] - Q0_var[141] - Q1_var[141]
   + total_stock_second_period_141 = 0
 R1417: total_stock_second_period_141 + Lost_1_expr_141
   = 511.4088725225142
 R1418: Sold_1[141] + Left_1[141] - total_stock_second_period_141 = 0
 R1419: - 700 Sold_0[141] - 700 Sold_1[141] + 700 Lost_0[141]
   + 700 Lost_1[141] + 300 Left_1[141] + profits_vars[141] = 0
 Fraction_lower_bound_142: Fraction[142] >= 0
 Fraction_upper_bound_142: Fraction[142] <= 1
 R1422: - 1.697734982775664 alphas[0] - 1.184042857768119 alphas[1]
   - 1.605197441370549 alphas[2] + 0.995543346039313 alphas[3] - alphas[4]
   + f_var[142] = 0
 R1423: - 2094.959652373807 Fraction[142] + Q0_var[142] = 0
 R1424: Q0_var[142] + Lost_0_expr_142 = 2090.234860729448
 R1425: Q0_var[142] + Q_hat_adjusted_142 = 2658.595025659854
 R1426: Sold_0[142] - Q0_var[142] - Q1_var[142]
   + total_stock_second_period_142 = 0
 R1427: total_stock_second_period_142 + Lost_1_expr_142
   = 568.3495539321655
 R1428: Sold_1[142] + Left_1[142] - total_stock_second_period_142 = 0
 R1429: - 700 Sold_0[142] - 700 Sold_1[142] + 700 Lost_0[142]
   + 700 Lost_1[142] + 300 Left_1[142] + profits_vars[142] = 0
 Fraction_lower_bound_143: Fraction[143] >= 0
 Fraction_upper_bound_143: Fraction[143] <= 1
 R1432: 0.2912144243363724 alphas[0] - 1.304365131214568 alphas[1]
   + 0.5845006399684045 alphas[2] - 1.668443127953897 alphas[3] - alphas[4]
   + f_var[143] = 0
 R1433: - 2094.959652373807 Fraction[143] + Q0_var[143] = 0
 R1434: Q0_var[143] + Lost_0_expr_143 = 1333.484681078238
 R1435: Q0_var[143] + Q_hat_adjusted_143 = 1590.477019223443
 R1436: Sold_0[143] - Q0_var[143] - Q1_var[143]
   + total_stock_second_period_143 = 0
 R1437: total_stock_second_period_143 + Lost_1_expr_143
   = 256.9817250913162
 R1438: Sold_1[143] + Left_1[143] - total_stock_second_period_143 = 0
 R1439: - 700 Sold_0[143] - 700 Sold_1[143] + 700 Lost_0[143]
   + 700 Lost_1[143] + 300 Left_1[143] + profits_vars[143] = 0
 Fraction_lower_bound_144: Fraction[144] >= 0
 Fraction_upper_bound_144: Fraction[144] <= 1
 R1442: 0.3575548497501934 alphas[0] - 0.098883718093063 alphas[1]
   + 1.056580708597242 alphas[2] - 1.576923555999606 alphas[3] - alphas[4]
   + f_var[144] = 0
 R1443: - 2094.959652373807 Fraction[144] + Q0_var[144] = 0
 R1444: Q0_var[144] + Lost_0_expr_144 = 767.4414383373116
 R1445: Q0_var[144] + Q_hat_adjusted_144 = 957.3055520433195
 R1446: Sold_0[144] - Q0_var[144] - Q1_var[144]
   + total_stock_second_period_144 = 0
 R1447: total_stock_second_period_144 + Lost_1_expr_144
   = 189.8534998531855
 R1448: Sold_1[144] + Left_1[144] - total_stock_second_period_144 = 0
 R1449: - 700 Sold_0[144] - 700 Sold_1[144] + 700 Lost_0[144]
   + 700 Lost_1[144] + 300 Left_1[144] + profits_vars[144] = 0
 Fraction_lower_bound_145: Fraction[145] >= 0
 Fraction_upper_bound_145: Fraction[145] <= 1
 R1452: 1.14998802938484 alphas[0] + 0.0628031561462819 alphas[1]
   + 1.497591218786749 alphas[2] + 0.7787327258374781 alphas[3] - alphas[4]
   + f_var[145] = 0
 R1453: - 2094.959652373807 Fraction[145] + Q0_var[145] = 0
 R1454: Q0_var[145] + Lost_0_expr_145 = 635.0377301562764
 R1455: Q0_var[145] + Q_hat_adjusted_145 = 762.191605795659
 R1456: Sold_0[145] - Q0_var[145] - Q1_var[145]
   + total_stock_second_period_145 = 0
 R1457: total_stock_second_period_145 + Lost_1_expr_145 = 127.143263120872
 R1458: Sold_1[145] + Left_1[145] - total_stock_second_period_145 = 0
 R1459: - 700 Sold_0[145] - 700 Sold_1[145] + 700 Lost_0[145]
   + 700 Lost_1[145] + 300 Left_1[145] + profits_vars[145] = 0
 Fraction_lower_bound_146: Fraction[146] >= 0
 Fraction_upper_bound_146: Fraction[146] <= 1
 R1462: - 0.6804969387014342 alphas[0] + 0.1332634361453927 alphas[1]
   - 0.729802932246047 alphas[2] + 1.058711065854552 alphas[3] - alphas[4]
   + f_var[146] = 0
 R1463: - 2094.959652373807 Fraction[146] + Q0_var[146] = 0
 R1464: Q0_var[146] + Lost_0_expr_146 = 1488.438331426776
 R1465: Q0_var[146] + Q_hat_adjusted_146 = 1932.320289723475
 R1466: Sold_0[146] - Q0_var[146] - Q1_var[146]
   + total_stock_second_period_146 = 0
 R1467: total_stock_second_period_146 + Lost_1_expr_146
   = 443.8713484625748
 R1468: Sold_1[146] + Left_1[146] - total_stock_second_period_146 = 0
 R1469: - 700 Sold_0[146] - 700 Sold_1[146] + 700 Lost_0[146]
   + 700 Lost_1[146] + 300 Left_1[146] + profits_vars[146] = 0
 Fraction_lower_bound_147: Fraction[147] >= 0
 Fraction_upper_bound_147: Fraction[147] <= 1
 R1472: - 0.5844872045801216 alphas[0] - 1.512703049605417 alphas[1]
   + 1.137975526789897 alphas[2] + 1.292345474596572 alphas[3] - alphas[4]
   + f_var[147] = 0
 R1473: - 2094.959652373807 Fraction[147] + Q0_var[147] = 0
 R1474: Q0_var[147] + Lost_0_expr_147 = 631.0970486458461
 R1475: Q0_var[147] + Q_hat_adjusted_147 = 809.3870889203114
 R1476: Sold_0[147] - Q0_var[147] - Q1_var[147]
   + total_stock_second_period_147 = 0
 R1477: total_stock_second_period_147 + Lost_1_expr_147
   = 178.2794257762002
 R1478: Sold_1[147] + Left_1[147] - total_stock_second_period_147 = 0
 R1479: - 700 Sold_0[147] - 700 Sold_1[147] + 700 Lost_0[147]
   + 700 Lost_1[147] + 300 Left_1[147] + profits_vars[147] = 0
 Fraction_lower_bound_148: Fraction[148] >= 0
 Fraction_upper_bound_148: Fraction[148] <= 1
 R1482: - 1.284136629937246 alphas[0] - 1.452204415815276 alphas[1]
   + 1.098338061095036 alphas[2] + 0.2889915035619232 alphas[3] - alphas[4]
   + f_var[148] = 0
 R1483: - 2094.959652373807 Fraction[148] + Q0_var[148] = 0
 R1484: Q0_var[148] + Lost_0_expr_148 = 859.5923337438492
 R1485: Q0_var[148] + Q_hat_adjusted_148 = 1043.5186912412
 R1486: Sold_0[148] - Q0_var[148] - Q1_var[148]
   + total_stock_second_period_148 = 0
 R1487: total_stock_second_period_148 + Lost_1_expr_148
   = 183.9157423590286
 R1488: Sold_1[148] + Left_1[148] - total_stock_second_period_148 = 0
 R1489: - 700 Sold_0[148] - 700 Sold_1[148] + 700 Lost_0[148]
   + 700 Lost_1[148] + 300 Left_1[148] + profits_vars[148] = 0
 Fraction_lower_bound_149: Fraction[149] >= 0
 Fraction_upper_bound_149: Fraction[149] <= 1
 R1492: 1.322261023738915 alphas[0] + 0.9520681236459169 alphas[1]
   + 0.8783560235914556 alphas[2] - 1.000248878252629 alphas[3] - alphas[4]
   + f_var[149] = 0
 R1493: - 2094.959652373807 Fraction[149] + Q0_var[149] = 0
 R1494: Q0_var[149] + Lost_0_expr_149 = 1241.228633023082
 R1495: Q0_var[149] + Q_hat_adjusted_149 = 1456.435707741619
 R1496: Sold_0[149] - Q0_var[149] - Q1_var[149]
   + total_stock_second_period_149 = 0
 R1497: total_stock_second_period_149 + Lost_1_expr_149
   = 215.1964609164928
 R1498: Sold_1[149] + Left_1[149] - total_stock_second_period_149 = 0
 R1499: - 700 Sold_0[149] - 700 Sold_1[149] + 700 Lost_0[149]
   + 700 Lost_1[149] + 300 Left_1[149] + profits_vars[149] = 0
 Fraction_lower_bound_150: Fraction[150] >= 0
 Fraction_upper_bound_150: Fraction[150] <= 1
 R1502: - 0.0426778818883356 alphas[0] + 1.355350614738002 alphas[1]
   + 0.7087805728885821 alphas[2] + 0.5081816226253278 alphas[3]
   - alphas[4] + f_var[150] = 0
 R1503: - 2094.959652373807 Fraction[150] + Q0_var[150] = 0
 R1504: Q0_var[150] + Lost_0_expr_150 = 799.5137118898315
 R1505: Q0_var[150] + Q_hat_adjusted_150 = 1038.833855087655
 R1506: Sold_0[150] - Q0_var[150] - Q1_var[150]
   + total_stock_second_period_150 = 0
 R1507: total_stock_second_period_150 + Lost_1_expr_150
   = 239.3095294532166
 R1508: Sold_1[150] + Left_1[150] - total_stock_second_period_150 = 0
 R1509: - 700 Sold_0[150] - 700 Sold_1[150] + 700 Lost_0[150]
   + 700 Lost_1[150] + 300 Left_1[150] + profits_vars[150] = 0
 Fraction_lower_bound_151: Fraction[151] >= 0
 Fraction_upper_bound_151: Fraction[151] <= 1
 R1512: - 0.3265752501557235 alphas[0] + 1.490142751590527 alphas[1]
   + 0.9374162489156025 alphas[2] + 0.8553706391877137 alphas[3]
   - alphas[4] + f_var[151] = 0
 R1513: - 2094.959652373807 Fraction[151] + Q0_var[151] = 0
 R1514: Q0_var[151] + Lost_0_expr_151 = 911.3885649941496
 R1515: Q0_var[151] + Q_hat_adjusted_151 = 1118.197471539935
 R1516: Sold_0[151] - Q0_var[151] - Q1_var[151]
   + total_stock_second_period_151 = 0
 R1517: total_stock_second_period_151 + Lost_1_expr_151
   = 206.7982921064874
 R1518: Sold_1[151] + Left_1[151] - total_stock_second_period_151 = 0
 R1519: - 700 Sold_0[151] - 700 Sold_1[151] + 700 Lost_0[151]
   + 700 Lost_1[151] + 300 Left_1[151] + profits_vars[151] = 0
 Fraction_lower_bound_152: Fraction[152] >= 0
 Fraction_upper_bound_152: Fraction[152] <= 1
 R1522: 0.8305005238017716 alphas[0] + 0.4834642311831147 alphas[1]
   - 0.4473580342014142 alphas[2] - 1.026446281014136 alphas[3] - alphas[4]
   + f_var[152] = 0
 R1523: - 2094.959652373807 Fraction[152] + Q0_var[152] = 0
 R1524: Q0_var[152] + Lost_0_expr_152 = 1615.231555467062
 R1525: Q0_var[152] + Q_hat_adjusted_152 = 2018.950784625879
 R1526: Sold_0[152] - Q0_var[152] - Q1_var[152]
   + total_stock_second_period_152 = 0
 R1527: total_stock_second_period_152 + Lost_1_expr_152
   = 403.7086176181698
 R1528: Sold_1[152] + Left_1[152] - total_stock_second_period_152 = 0
 R1529: - 700 Sold_0[152] - 700 Sold_1[152] + 700 Lost_0[152]
   + 700 Lost_1[152] + 300 Left_1[152] + profits_vars[152] = 0
 Fraction_lower_bound_153: Fraction[153] >= 0
 Fraction_upper_bound_153: Fraction[153] <= 1
 R1532: 1.077682980836419 alphas[0] + 0.2847060894666428 alphas[1]
   - 0.1087380251547212 alphas[2] + 0.3839307718031325 alphas[3]
   - alphas[4] + f_var[153] = 0
 R1533: - 2094.959652373807 Fraction[153] + Q0_var[153] = 0
 R1534: Q0_var[153] + Lost_0_expr_153 = 1829.154445955313
 R1535: Q0_var[153] + Q_hat_adjusted_153 = 2184.723029566814
 R1536: Sold_0[153] - Q0_var[153] - Q1_var[153]
   + total_stock_second_period_153 = 0
 R1537: total_stock_second_period_153 + Lost_1_expr_153
   = 355.5579716832476
 R1538: Sold_1[153] + Left_1[153] - total_stock_second_period_153 = 0
 R1539: - 700 Sold_0[153] - 700 Sold_1[153] + 700 Lost_0[153]
   + 700 Lost_1[153] + 300 Left_1[153] + profits_vars[153] = 0
 Fraction_lower_bound_154: Fraction[154] >= 0
 Fraction_upper_bound_154: Fraction[154] <= 1
 R1542: - 1.275678976896877 alphas[0] + 0.848546321573721 alphas[1]
   - 0.5995041739198846 alphas[2] + 1.514639135022048 alphas[3] - alphas[4]
   + f_var[154] = 0
 R1543: - 2094.959652373807 Fraction[154] + Q0_var[154] = 0
 R1544: Q0_var[154] + Lost_0_expr_154 = 1470.639677097006
 R1545: Q0_var[154] + Q_hat_adjusted_154 = 1895.993583212256
 R1546: Sold_0[154] - Q0_var[154] - Q1_var[154]
   + total_stock_second_period_154 = 0
 R1547: total_stock_second_period_154 + Lost_1_expr_154
   = 425.3432957477161
 R1548: Sold_1[154] + Left_1[154] - total_stock_second_period_154 = 0
 R1549: - 700 Sold_0[154] - 700 Sold_1[154] + 700 Lost_0[154]
   + 700 Lost_1[154] + 300 Left_1[154] + profits_vars[154] = 0
 Fraction_lower_bound_155: Fraction[155] >= 0
 Fraction_upper_bound_155: Fraction[155] <= 1
 R1552: 1.454226398035204 alphas[0] - 1.383848026654478 alphas[1]
   - 0.5691294311632378 alphas[2] + 0.1660988233584148 alphas[3]
   - alphas[4] + f_var[155] = 0
 R1553: - 2094.959652373807 Fraction[155] + Q0_var[155] = 0
 R1554: Q0_var[155] + Lost_0_expr_155 = 1584.510587266324
 R1555: Q0_var[155] + Q_hat_adjusted_155 = 2005.545306522693
 R1556: Sold_0[155] - Q0_var[155] - Q1_var[155]
   + total_stock_second_period_155 = 0
 R1557: total_stock_second_period_155 + Lost_1_expr_155
   = 421.0241077352973
 R1558: Sold_1[155] + Left_1[155] - total_stock_second_period_155 = 0
 R1559: - 700 Sold_0[155] - 700 Sold_1[155] + 700 Lost_0[155]
   + 700 Lost_1[155] + 300 Left_1[155] + profits_vars[155] = 0
 Fraction_lower_bound_156: Fraction[156] >= 0
 Fraction_upper_bound_156: Fraction[156] <= 1
 R1562: 0.0498677758766646 alphas[0] - 1.108110061483355 alphas[1]
   - 1.318470130163492 alphas[2] - 1.306833481848668 alphas[3] - alphas[4]
   + f_var[156] = 0
 R1563: - 2094.959652373807 Fraction[156] + Q0_var[156] = 0
 R1564: Q0_var[156] + Lost_0_expr_156 = 2091.059538069346
 R1565: Q0_var[156] + Q_hat_adjusted_156 = 2618.648025761492
 R1566: Sold_0[156] - Q0_var[156] - Q1_var[156]
   + total_stock_second_period_156 = 0
 R1567: total_stock_second_period_156 + Lost_1_expr_156
   = 527.5778780921252
 R1568: Sold_1[156] + Left_1[156] - total_stock_second_period_156 = 0
 R1569: - 700 Sold_0[156] - 700 Sold_1[156] + 700 Lost_0[156]
   + 700 Lost_1[156] + 300 Left_1[156] + profits_vars[156] = 0
 Fraction_lower_bound_157: Fraction[157] >= 0
 Fraction_upper_bound_157: Fraction[157] <= 1
 R1572: 1.258955364066543 alphas[0] - 0.8004408689951856 alphas[1]
   + 1.559336225168714 alphas[2] - 1.759393421035008 alphas[3] - alphas[4]
   + f_var[157] = 0
 R1573: - 2094.959652373807 Fraction[157] + Q0_var[157] = 0
 R1574: Q0_var[157] + Lost_0_expr_157 = 759.3635603841593
 R1575: Q0_var[157] + Q_hat_adjusted_157 = 877.7375034821991
 R1576: Sold_0[157] - Q0_var[157] - Q1_var[157]
   + total_stock_second_period_157 = 0
 R1577: total_stock_second_period_157 + Lost_1_expr_157
   = 118.3633260288806
 R1578: Sold_1[157] + Left_1[157] - total_stock_second_period_157 = 0
 R1579: - 700 Sold_0[157] - 700 Sold_1[157] + 700 Lost_0[157]
   + 700 Lost_1[157] + 300 Left_1[157] + profits_vars[157] = 0
 Fraction_lower_bound_158: Fraction[158] >= 0
 Fraction_upper_bound_158: Fraction[158] <= 1
 R1582: 0.0960865572067635 alphas[0] + 0.0644586304607989 alphas[1]
   - 0.2725805002077354 alphas[2] - 0.3069626079505597 alphas[3]
   - alphas[4] + f_var[158] = 0
 R1583: - 2094.959652373807 Fraction[158] + Q0_var[158] = 0
 R1584: Q0_var[158] + Lost_0_expr_158 = 1470.373620015705
 R1585: Q0_var[158] + Q_hat_adjusted_158 = 1849.240061611566
 R1586: Sold_0[158] - Q0_var[158] - Q1_var[158]
   + total_stock_second_period_158 = 0
 R1587: total_stock_second_period_158 + Lost_1_expr_158
   = 378.8558300336417
 R1588: Sold_1[158] + Left_1[158] - total_stock_second_period_158 = 0
 R1589: - 700 Sold_0[158] - 700 Sold_1[158] + 700 Lost_0[158]
   + 700 Lost_1[158] + 300 Left_1[158] + profits_vars[158] = 0
 Fraction_lower_bound_159: Fraction[159] >= 0
 Fraction_upper_bound_159: Fraction[159] <= 1
 R1592: - 1.685930212891279 alphas[0] - 1.27189027900331 alphas[1]
   + 0.3314070641922182 alphas[2] + 0.3345987460364157 alphas[3]
   - alphas[4] + f_var[159] = 0
 R1593: - 2094.959652373807 Fraction[159] + Q0_var[159] = 0
 R1594: Q0_var[159] + Lost_0_expr_159 = 1323.733607878937
 R1595: Q0_var[159] + Q_hat_adjusted_159 = 1616.715014748603
 R1596: Sold_0[159] - Q0_var[159] - Q1_var[159]
   + total_stock_second_period_159 = 0
 R1597: total_stock_second_period_159 + Lost_1_expr_159
   = 292.9707949336987
 R1598: Sold_1[159] + Left_1[159] - total_stock_second_period_159 = 0
 R1599: - 700 Sold_0[159] - 700 Sold_1[159] + 700 Lost_0[159]
   + 700 Lost_1[159] + 300 Left_1[159] + profits_vars[159] = 0
 Fraction_lower_bound_160: Fraction[160] >= 0
 Fraction_upper_bound_160: Fraction[160] <= 1
 R1602: 0.2097491025304893 alphas[0] - 1.466705381600829 alphas[1]
   - 1.086780176597796 alphas[2] - 0.2472543471491074 alphas[3] - alphas[4]
   + f_var[160] = 0
 R1603: - 2094.959652373807 Fraction[160] + Q0_var[160] = 0
 R1604: Q0_var[160] + Lost_0_expr_160 = 2206.964853220385
 R1605: Q0_var[160] + Q_hat_adjusted_160 = 2701.607798570979
 R1606: Sold_0[160] - Q0_var[160] - Q1_var[160]
   + total_stock_second_period_160 = 0
 R1607: total_stock_second_period_160 + Lost_1_expr_160
   = 494.6323325756994
 R1608: Sold_1[160] + Left_1[160] - total_stock_second_period_160 = 0
 R1609: - 700 Sold_0[160] - 700 Sold_1[160] + 700 Lost_0[160]
   + 700 Lost_1[160] + 300 Left_1[160] + profits_vars[160] = 0
 Fraction_lower_bound_161: Fraction[161] >= 0
 Fraction_upper_bound_161: Fraction[161] <= 1
 R1612: - 1.267309386430352 alphas[0] - 1.275794547359101 alphas[1]
   + 0.8848195129328814 alphas[2] + 0.3801100346704351 alphas[3]
   - alphas[4] + f_var[161] = 0
 R1613: - 2094.959652373807 Fraction[161] + Q0_var[161] = 0
 R1614: Q0_var[161] + Lost_0_expr_161 = 759.550583475306
 R1615: Q0_var[161] + Q_hat_adjusted_161 = 973.8385707677356
 R1616: Sold_0[161] - Q0_var[161] - Q1_var[161]
   + total_stock_second_period_161 = 0
 R1617: total_stock_second_period_161 + Lost_1_expr_161
   = 214.2773740980794
 R1618: Sold_1[161] + Left_1[161] - total_stock_second_period_161 = 0
 R1619: - 700 Sold_0[161] - 700 Sold_1[161] + 700 Lost_0[161]
   + 700 Lost_1[161] + 300 Left_1[161] + profits_vars[161] = 0
 Fraction_lower_bound_162: Fraction[162] >= 0
 Fraction_upper_bound_162: Fraction[162] <= 1
 R1622: 1.253906307072308 alphas[0] + 0.803120068425258 alphas[1]
   - 0.4527536720404967 alphas[2] + 0.7188318883832582 alphas[3]
   - alphas[4] + f_var[162] = 0
 R1623: - 2094.959652373807 Fraction[162] + Q0_var[162] = 0
 R1624: Q0_var[162] + Lost_0_expr_162 = 1457.705769615989
 R1625: Q0_var[162] + Q_hat_adjusted_162 = 1862.192240719266
 R1626: Sold_0[162] - Q0_var[162] - Q1_var[162]
   + total_stock_second_period_162 = 0
 R1627: total_stock_second_period_162 + Lost_1_expr_162 = 404.47585948292
 R1628: Sold_1[162] + Left_1[162] - total_stock_second_period_162 = 0
 R1629: - 700 Sold_0[162] - 700 Sold_1[162] + 700 Lost_0[162]
   + 700 Lost_1[162] + 300 Left_1[162] + profits_vars[162] = 0
 Fraction_lower_bound_163: Fraction[163] >= 0
 Fraction_upper_bound_163: Fraction[163] <= 1
 R1632: 0.7293020295324172 alphas[0] - 1.197974751971166 alphas[1]
   + 0.7405170169855209 alphas[2] - 0.8651870607475508 alphas[3]
   - alphas[4] + f_var[163] = 0
 R1633: - 2094.959652373807 Fraction[163] + Q0_var[163] = 0
 R1634: Q0_var[163] + Lost_0_expr_163 = 763.0501638787257
 R1635: Q0_var[163] + Q_hat_adjusted_163 = 997.8574897984917
 R1636: Sold_0[163] - Q0_var[163] - Q1_var[163]
   + total_stock_second_period_163 = 0
 R1637: total_stock_second_period_163 + Lost_1_expr_163
   = 234.7967120117548
 R1638: Sold_1[163] + Left_1[163] - total_stock_second_period_163 = 0
 R1639: - 700 Sold_0[163] - 700 Sold_1[163] + 700 Lost_0[163]
   + 700 Lost_1[163] + 300 Left_1[163] + profits_vars[163] = 0
 Fraction_lower_bound_164: Fraction[164] >= 0
 Fraction_upper_bound_164: Fraction[164] <= 1
 R1642: 0.2776132257000437 alphas[0] + 0.5076826564099798 alphas[1]
   + 0.8633125550482307 alphas[2] + 0.9976025068511383 alphas[3]
   - alphas[4] + f_var[164] = 0
 R1643: - 2094.959652373807 Fraction[164] + Q0_var[164] = 0
 R1644: Q0_var[164] + Lost_0_expr_164 = 800.4750615408716
 R1645: Q0_var[164] + Q_hat_adjusted_164 = 1017.821267273144
 R1646: Sold_0[164] - Q0_var[164] - Q1_var[164]
   + total_stock_second_period_164 = 0
 R1647: total_stock_second_period_164 + Lost_1_expr_164
   = 217.3355924152405
 R1648: Sold_1[164] + Left_1[164] - total_stock_second_period_164 = 0
 R1649: - 700 Sold_0[164] - 700 Sold_1[164] + 700 Lost_0[164]
   + 700 Lost_1[164] + 300 Left_1[164] + profits_vars[164] = 0
 Fraction_lower_bound_165: Fraction[165] >= 0
 Fraction_upper_bound_165: Fraction[165] <= 1
 R1652: 0.291178819056588 alphas[0] + 1.261492457361713 alphas[1]
   + 0.5357177377288574 alphas[2] + 1.326510001230115 alphas[3] - alphas[4]
   + f_var[165] = 0
 R1653: - 2094.959652373807 Fraction[165] + Q0_var[165] = 0
 R1654: Q0_var[165] + Lost_0_expr_165 = 676.6640105223669
 R1655: Q0_var[165] + Q_hat_adjusted_165 = 940.5931159178228
 R1656: Sold_0[165] - Q0_var[165] - Q1_var[165]
   + total_stock_second_period_165 = 0
 R1657: total_stock_second_period_165 + Lost_1_expr_165
   = 263.9184925317116
 R1658: Sold_1[165] + Left_1[165] - total_stock_second_period_165 = 0
 R1659: - 700 Sold_0[165] - 700 Sold_1[165] + 700 Lost_0[165]
   + 700 Lost_1[165] + 300 Left_1[165] + profits_vars[165] = 0
 Fraction_lower_bound_166: Fraction[166] >= 0
 Fraction_upper_bound_166: Fraction[166] <= 1
 R1662: - 0.6343162667560971 alphas[0] - 0.7976504152501533 alphas[1]
   + 1.464927957989507 alphas[2] - 1.169023204606428 alphas[3] - alphas[4]
   + f_var[166] = 0
 R1663: - 2094.959652373807 Fraction[166] + Q0_var[166] = 0
 R1664: Q0_var[166] + Lost_0_expr_166 = 542.5703198251656
 R1665: Q0_var[166] + Q_hat_adjusted_166 = 674.3688041310656
 R1666: Sold_0[166] - Q0_var[166] - Q1_var[166]
   + total_stock_second_period_166 = 0
 R1667: total_stock_second_period_166 + Lost_1_expr_166
   = 131.7878706435609
 R1668: Sold_1[166] + Left_1[166] - total_stock_second_period_166 = 0
 R1669: - 700 Sold_0[166] - 700 Sold_1[166] + 700 Lost_0[166]
   + 700 Lost_1[166] + 300 Left_1[166] + profits_vars[166] = 0
 Fraction_lower_bound_167: Fraction[167] >= 0
 Fraction_upper_bound_167: Fraction[167] <= 1
 R1672: 0.4789351493149436 alphas[0] - 1.014275239712 alphas[1]
   - 0.9360086081418874 alphas[2] + 1.547546570196807 alphas[3] - alphas[4]
   + f_var[167] = 0
 R1673: - 2094.959652373807 Fraction[167] + Q0_var[167] = 0
 R1674: Q0_var[167] + Lost_0_expr_167 = 1739.078625978894
 R1675: Q0_var[167] + Q_hat_adjusted_167 = 2212.28235127089
 R1676: Sold_0[167] - Q0_var[167] - Q1_var[167]
   + total_stock_second_period_167 = 0
 R1677: total_stock_second_period_167 + Lost_1_expr_167
   = 473.1931139358691
 R1678: Sold_1[167] + Left_1[167] - total_stock_second_period_167 = 0
 R1679: - 700 Sold_0[167] - 700 Sold_1[167] + 700 Lost_0[167]
   + 700 Lost_1[167] + 300 Left_1[167] + profits_vars[167] = 0
 Fraction_lower_bound_168: Fraction[168] >= 0
 Fraction_upper_bound_168: Fraction[168] <= 1
 R1682: - 0.7787557319727031 alphas[0] - 1.604766463100093 alphas[1]
   - 1.410435363257428 alphas[2] + 1.433602556392124 alphas[3] - alphas[4]
   + f_var[168] = 0
 R1683: - 2094.959652373807 Fraction[168] + Q0_var[168] = 0
 R1684: Q0_var[168] + Lost_0_expr_168 = 1788.198082364279
 R1685: Q0_var[168] + Q_hat_adjusted_168 = 2328.863722696205
 R1686: Sold_0[168] - Q0_var[168] - Q1_var[168]
   + total_stock_second_period_168 = 0
 R1687: total_stock_second_period_168 + Lost_1_expr_168
   = 540.6550302353056
 R1688: Sold_1[168] + Left_1[168] - total_stock_second_period_168 = 0
 R1689: - 700 Sold_0[168] - 700 Sold_1[168] + 700 Lost_0[168]
   + 700 Lost_1[168] + 300 Left_1[168] + profits_vars[168] = 0
 Fraction_lower_bound_169: Fraction[169] >= 0
 Fraction_upper_bound_169: Fraction[169] <= 1
 R1692: - 0.5245926994494621 alphas[0] + 0.0742137694746016 alphas[1]
   + 1.628752554502461 alphas[2] + 0.1905631530628615 alphas[3] - alphas[4]
   + f_var[169] = 0
 R1693: - 2094.959652373807 Fraction[169] + Q0_var[169] = 0
 R1694: Q0_var[169] + Lost_0_expr_169 = 744.5910617144899
 R1695: Q0_var[169] + Q_hat_adjusted_169 = 853.0942314066133
 R1696: Sold_0[169] - Q0_var[169] - Q1_var[169]
   + total_stock_second_period_169 = 0
 R1697: total_stock_second_period_169 + Lost_1_expr_169
   = 108.4925547459586
 R1698: Sold_1[169] + Left_1[169] - total_stock_second_period_169 = 0
 R1699: - 700 Sold_0[169] - 700 Sold_1[169] + 700 Lost_0[169]
   + 700 Lost_1[169] + 300 Left_1[169] + profits_vars[169] = 0
 Fraction_lower_bound_170: Fraction[170] >= 0
 Fraction_upper_bound_170: Fraction[170] <= 1
 R1702: 0.2933973611501832 alphas[0] + 0.8374373015996261 alphas[1]
   - 0.6056780995983352 alphas[2] - 1.801027198358978 alphas[3] - alphas[4]
   + f_var[170] = 0
 R1703: - 2094.959652373807 Fraction[170] + Q0_var[170] = 0
 R1704: Q0_var[170] + Lost_0_expr_170 = 1767.838506631978
 R1705: Q0_var[170] + Q_hat_adjusted_170 = 2194.070325095018
 R1706: Sold_0[170] - Q0_var[170] - Q1_var[170]
   + total_stock_second_period_170 = 0
 R1707: total_stock_second_period_170 + Lost_1_expr_170 = 426.221207570495
 R1708: Sold_1[170] + Left_1[170] - total_stock_second_period_170 = 0
 R1709: - 700 Sold_0[170] - 700 Sold_1[170] + 700 Lost_0[170]
   + 700 Lost_1[170] + 300 Left_1[170] + profits_vars[170] = 0
 Fraction_lower_bound_171: Fraction[171] >= 0
 Fraction_upper_bound_171: Fraction[171] <= 1
 R1712: 0.1823024342626471 alphas[0] + 1.453602713420972 alphas[1]
   - 0.2908263633722077 alphas[2] + 1.35785270721478 alphas[3] - alphas[4]
   + f_var[171] = 0
 R1713: - 2094.959652373807 Fraction[171] + Q0_var[171] = 0
 R1714: Q0_var[171] + Lost_0_expr_171 = 1545.099074270043
 R1715: Q0_var[171] + Q_hat_adjusted_171 = 1926.560016547309
 R1716: Sold_0[171] - Q0_var[171] - Q1_var[171]
   + total_stock_second_period_171 = 0
 R1717: total_stock_second_period_171 + Lost_1_expr_171
   = 381.4503314616078
 R1718: Sold_1[171] + Left_1[171] - total_stock_second_period_171 = 0
 R1719: - 700 Sold_0[171] - 700 Sold_1[171] + 700 Lost_0[171]
   + 700 Lost_1[171] + 300 Left_1[171] + profits_vars[171] = 0
 Fraction_lower_bound_172: Fraction[172] >= 0
 Fraction_upper_bound_172: Fraction[172] <= 1
 R1722: - 0.4405559671813802 alphas[0] + 0.2991758618453004 alphas[1]
   - 1.495120312697724 alphas[2] + 0.3939744329225515 alphas[3] - alphas[4]
   + f_var[172] = 0
 R1723: - 2094.959652373807 Fraction[172] + Q0_var[172] = 0
 R1724: Q0_var[172] + Lost_0_expr_172 = 2237.910684109235
 R1725: Q0_var[172] + Q_hat_adjusted_172 = 2790.618242836678
 R1726: Sold_0[172] - Q0_var[172] - Q1_var[172]
   + total_stock_second_period_172 = 0
 R1727: total_stock_second_period_172 + Lost_1_expr_172
   = 552.6969501088421
 R1728: Sold_1[172] + Left_1[172] - total_stock_second_period_172 = 0
 R1729: - 700 Sold_0[172] - 700 Sold_1[172] + 700 Lost_0[172]
   + 700 Lost_1[172] + 300 Left_1[172] + profits_vars[172] = 0
 Fraction_lower_bound_173: Fraction[173] >= 0
 Fraction_upper_bound_173: Fraction[173] <= 1
 R1732: 1.415010477999216 alphas[0] - 0.2460030841422325 alphas[1]
   - 0.7214255568066906 alphas[2] + 1.386686130967184 alphas[3] - alphas[4]
   + f_var[173] = 0
 R1733: - 2094.959652373807 Fraction[173] + Q0_var[173] = 0
 R1734: Q0_var[173] + Lost_0_expr_173 = 2171.079233521988
 R1735: Q0_var[173] + Q_hat_adjusted_173 = 2613.769958686129
 R1736: Sold_0[173] - Q0_var[173] - Q1_var[173]
   + total_stock_second_period_173 = 0
 R1737: total_stock_second_period_173 + Lost_1_expr_173
   = 442.6801133675785
 R1738: Sold_1[173] + Left_1[173] - total_stock_second_period_173 = 0
 R1739: - 700 Sold_0[173] - 700 Sold_1[173] + 700 Lost_0[173]
   + 700 Lost_1[173] + 300 Left_1[173] + profits_vars[173] = 0
 Fraction_lower_bound_174: Fraction[174] >= 0
 Fraction_upper_bound_174: Fraction[174] <= 1
 R1742: - 1.148992393373951 alphas[0] + 0.0226298307952906 alphas[1]
   + 0.2166245603902661 alphas[2] + 0.1755411518869472 alphas[3]
   - alphas[4] + f_var[174] = 0
 R1743: - 2094.959652373807 Fraction[174] + Q0_var[174] = 0
 R1744: Q0_var[174] + Lost_0_expr_174 = 1086.354929764516
 R1745: Q0_var[174] + Q_hat_adjusted_174 = 1395.658029290691
 R1746: Sold_0[174] - Q0_var[174] - Q1_var[174]
   + total_stock_second_period_174 = 0
 R1747: total_stock_second_period_174 + Lost_1_expr_174 = 309.292487546168
 R1748: Sold_1[174] + Left_1[174] - total_stock_second_period_174 = 0
 R1749: - 700 Sold_0[174] - 700 Sold_1[174] + 700 Lost_0[174]
   + 700 Lost_1[174] + 300 Left_1[174] + profits_vars[174] = 0
 Fraction_lower_bound_175: Fraction[175] >= 0
 Fraction_upper_bound_175: Fraction[175] <= 1
 R1752: - 0.5731020137707143 alphas[0] + 1.1303155461177 alphas[1]
   - 0.3446088411280799 alphas[2] + 0.5287954284249047 alphas[3]
   - alphas[4] + f_var[175] = 0
 R1753: - 2094.959652373807 Fraction[175] + Q0_var[175] = 0
 R1754: Q0_var[175] + Lost_0_expr_175 = 1679.069769820995
 R1755: Q0_var[175] + Q_hat_adjusted_175 = 2068.178403058836
 R1756: Sold_0[175] - Q0_var[175] - Q1_var[175]
   + total_stock_second_period_175 = 0
 R1757: total_stock_second_period_175 + Lost_1_expr_175
   = 389.0980220136706
 R1758: Sold_1[175] + Left_1[175] - total_stock_second_period_175 = 0
 R1759: - 700 Sold_0[175] - 700 Sold_1[175] + 700 Lost_0[175]
   + 700 Lost_1[175] + 300 Left_1[175] + profits_vars[175] = 0
 Fraction_lower_bound_176: Fraction[176] >= 0
 Fraction_upper_bound_176: Fraction[176] <= 1
 R1762: - 0.8216124353486739 alphas[0] + 0.4373852880091484 alphas[1]
   + 1.241157838247586 alphas[2] - 0.4862184068297812 alphas[3] - alphas[4]
   + f_var[176] = 0
 R1763: - 2094.959652373807 Fraction[176] + Q0_var[176] = 0
 R1764: Q0_var[176] + Lost_0_expr_176 = 615.1309393411314
 R1765: Q0_var[176] + Q_hat_adjusted_176 = 778.7487958167733
 R1766: Sold_0[176] - Q0_var[176] - Q1_var[176]
   + total_stock_second_period_176 = 0
 R1767: total_stock_second_period_176 + Lost_1_expr_176
   = 163.6072421599445
 R1768: Sold_1[176] + Left_1[176] - total_stock_second_period_176 = 0
 R1769: - 700 Sold_0[176] - 700 Sold_1[176] + 700 Lost_0[176]
   + 700 Lost_1[176] + 300 Left_1[176] + profits_vars[176] = 0
 Fraction_lower_bound_177: Fraction[177] >= 0
 Fraction_upper_bound_177: Fraction[177] <= 1
 R1772: - 0.176084879536981 alphas[0] - 1.37615180342689 alphas[1]
   + 1.118315193876172 alphas[2] - 1.646180823508078 alphas[3] - alphas[4]
   + f_var[177] = 0
 R1773: - 2094.959652373807 Fraction[177] + Q0_var[177] = 0
 R1774: Q0_var[177] + Lost_0_expr_177 = 642.6433376589987
 R1775: Q0_var[177] + Q_hat_adjusted_177 = 823.7290110098796
 R1776: Sold_0[177] - Q0_var[177] - Q1_var[177]
   + total_stock_second_period_177 = 0
 R1777: total_stock_second_period_177 + Lost_1_expr_177
   = 181.0750601305231
 R1778: Sold_1[177] + Left_1[177] - total_stock_second_period_177 = 0
 R1779: - 700 Sold_0[177] - 700 Sold_1[177] + 700 Lost_0[177]
   + 700 Lost_1[177] + 300 Left_1[177] + profits_vars[177] = 0
 Fraction_lower_bound_178: Fraction[178] >= 0
 Fraction_upper_bound_178: Fraction[178] <= 1
 R1782: 1.277211251012474 alphas[0] + 0.1461218805049531 alphas[1]
   + 0.9014085127194295 alphas[2] + 1.038873489923543 alphas[3] - alphas[4]
   + f_var[178] = 0
 R1783: - 2094.959652373807 Fraction[178] + Q0_var[178] = 0
 R1784: Q0_var[178] + Lost_0_expr_178 = 1228.080107464607
 R1785: Q0_var[178] + Q_hat_adjusted_178 = 1440.009194685653
 R1786: Sold_0[178] - Q0_var[178] - Q1_var[178]
   + total_stock_second_period_178 = 0
 R1787: total_stock_second_period_178 + Lost_1_expr_178
   = 211.9184731654302
 R1788: Sold_1[178] + Left_1[178] - total_stock_second_period_178 = 0
 R1789: - 700 Sold_0[178] - 700 Sold_1[178] + 700 Lost_0[178]
   + 700 Lost_1[178] + 300 Left_1[178] + profits_vars[178] = 0
 Fraction_lower_bound_179: Fraction[179] >= 0
 Fraction_upper_bound_179: Fraction[179] <= 1
 R1792: 0.2733776074765876 alphas[0] - 0.0295673998256222 alphas[1]
   - 0.0101919607438187 alphas[2] - 0.6687053360177392 alphas[3]
   - alphas[4] + f_var[179] = 0
 R1793: - 2094.959652373807 Fraction[179] + Q0_var[179] = 0
 R1794: Q0_var[179] + Lost_0_expr_179 = 1274.396032361624
 R1795: Q0_var[179] + Q_hat_adjusted_179 = 1615.951691714579
 R1796: Sold_0[179] - Q0_var[179] - Q1_var[179]
   + total_stock_second_period_179 = 0
 R1797: total_stock_second_period_179 + Lost_1_expr_179
   = 341.5450470805698
 R1798: Sold_1[179] + Left_1[179] - total_stock_second_period_179 = 0
 R1799: - 700 Sold_0[179] - 700 Sold_1[179] + 700 Lost_0[179]
   + 700 Lost_1[179] + 300 Left_1[179] + profits_vars[179] = 0
 Fraction_lower_bound_180: Fraction[180] >= 0
 Fraction_upper_bound_180: Fraction[180] <= 1
 R1802: 0.2722258321798704 alphas[0] + 0.1042397285346091 alphas[1]
   + 0.7502546319836576 alphas[2] - 1.619927505429584 alphas[3] - alphas[4]
   + f_var[180] = 0
 R1803: - 2094.959652373807 Fraction[180] + Q0_var[180] = 0
 R1804: Q0_var[180] + Lost_0_expr_180 = 1048.111424598424
 R1805: Q0_var[180] + Q_hat_adjusted_180 = 1281.534093186844
 R1806: Sold_0[180] - Q0_var[180] - Q1_var[180]
   + total_stock_second_period_180 = 0
 R1807: total_stock_second_period_180 + Lost_1_expr_180
   = 233.4120553476735
 R1808: Sold_1[180] + Left_1[180] - total_stock_second_period_180 = 0
 R1809: - 700 Sold_0[180] - 700 Sold_1[180] + 700 Lost_0[180]
   + 700 Lost_1[180] + 300 Left_1[180] + profits_vars[180] = 0
 Fraction_lower_bound_181: Fraction[181] >= 0
 Fraction_upper_bound_181: Fraction[181] <= 1
 R1812: 0.5596182706083201 alphas[0] + 0.393670769331236 alphas[1]
   + 1.334936726148082 alphas[2] - 1.088108578083917 alphas[3] - alphas[4]
   + f_var[181] = 0
 R1813: - 2094.959652373807 Fraction[181] + Q0_var[181] = 0
 R1814: Q0_var[181] + Lost_0_expr_181 = 606.5252479629278
 R1815: Q0_var[181] + Q_hat_adjusted_181 = 756.8080574812136
 R1816: Sold_0[181] - Q0_var[181] - Q1_var[181]
   + total_stock_second_period_181 = 0
 R1817: total_stock_second_period_181 + Lost_1_expr_181
   = 150.2721942986757
 R1818: Sold_1[181] + Left_1[181] - total_stock_second_period_181 = 0
 R1819: - 700 Sold_0[181] - 700 Sold_1[181] + 700 Lost_0[181]
   + 700 Lost_1[181] + 300 Left_1[181] + profits_vars[181] = 0
 Fraction_lower_bound_182: Fraction[182] >= 0
 Fraction_upper_bound_182: Fraction[182] <= 1
 R1822: 1.551640596787007 alphas[0] + 1.469417294961654 alphas[1]
   - 0.1347319583724882 alphas[2] - 1.325972433975404 alphas[3] - alphas[4]
   + f_var[182] = 0
 R1823: - 2094.959652373807 Fraction[182] + Q0_var[182] = 0
 R1824: Q0_var[182] + Lost_0_expr_182 = 1548.402021008132
 R1825: Q0_var[182] + Q_hat_adjusted_182 = 1907.666856041747
 R1826: Sold_0[182] - Q0_var[182] - Q1_var[182]
   + total_stock_second_period_182 = 0
 R1827: total_stock_second_period_182 + Lost_1_expr_182 = 359.254223060561
 R1828: Sold_1[182] + Left_1[182] - total_stock_second_period_182 = 0
 R1829: - 700 Sold_0[182] - 700 Sold_1[182] + 700 Lost_0[182]
   + 700 Lost_1[182] + 300 Left_1[182] + profits_vars[182] = 0
 Fraction_lower_bound_183: Fraction[183] >= 0
 Fraction_upper_bound_183: Fraction[183] <= 1
 R1832: - 0.8587989703181248 alphas[0] + 0.9099243191460212 alphas[1]
   - 0.7652716685141694 alphas[2] + 0.2503958775281196 alphas[3]
   - alphas[4] + f_var[183] = 0
 R1833: - 2094.959652373807 Fraction[183] + Q0_var[183] = 0
 R1834: Q0_var[183] + Lost_0_expr_183 = 1966.01250220268
 R1835: Q0_var[183] + Q_hat_adjusted_183 = 2414.937999569125
 R1836: Sold_0[183] - Q0_var[183] - Q1_var[183]
   + total_stock_second_period_183 = 0
 R1837: total_stock_second_period_183 + Lost_1_expr_183
   = 448.9148855134819
 R1838: Sold_1[183] + Left_1[183] - total_stock_second_period_183 = 0
 R1839: - 700 Sold_0[183] - 700 Sold_1[183] + 700 Lost_0[183]
   + 700 Lost_1[183] + 300 Left_1[183] + profits_vars[183] = 0
 Fraction_lower_bound_184: Fraction[184] >= 0
 Fraction_upper_bound_184: Fraction[184] <= 1
 R1842: 1.279571761091889 alphas[0] + 0.0110130897099541 alphas[1]
   + 1.349372034400004 alphas[2] - 1.705858513896594 alphas[3] - alphas[4]
   + f_var[184] = 0
 R1843: - 2094.959652373807 Fraction[184] + Q0_var[184] = 0
 R1844: Q0_var[184] + Lost_0_expr_184 = 582.4441723661721
 R1845: Q0_var[184] + Q_hat_adjusted_184 = 730.6743279052371
 R1846: Sold_0[184] - Q0_var[184] - Q1_var[184]
   + total_stock_second_period_184 = 0
 R1847: total_stock_second_period_184 + Lost_1_expr_184 = 148.219541181427
 R1848: Sold_1[184] + Left_1[184] - total_stock_second_period_184 = 0
 R1849: - 700 Sold_0[184] - 700 Sold_1[184] + 700 Lost_0[184]
   + 700 Lost_1[184] + 300 Left_1[184] + profits_vars[184] = 0
 Fraction_lower_bound_185: Fraction[185] >= 0
 Fraction_upper_bound_185: Fraction[185] <= 1
 R1852: - 0.4125442900793445 alphas[0] - 0.2692521581624105 alphas[1]
   - 0.874708278752015 alphas[2] + 0.9135762003707996 alphas[3] - alphas[4]
   + f_var[185] = 0
 R1853: - 2094.959652373807 Fraction[185] + Q0_var[185] = 0
 R1854: Q0_var[185] + Lost_0_expr_185 = 2041.448617544953
 R1855: Q0_var[185] + Q_hat_adjusted_185 = 2505.935637679462
 R1856: Sold_0[185] - Q0_var[185] - Q1_var[185]
   + total_stock_second_period_185 = 0
 R1857: total_stock_second_period_185 + Lost_1_expr_185
   = 464.4764097326664
 R1858: Sold_1[185] + Left_1[185] - total_stock_second_period_185 = 0
 R1859: - 700 Sold_0[185] - 700 Sold_1[185] + 700 Lost_0[185]
   + 700 Lost_1[185] + 300 Left_1[185] + profits_vars[185] = 0
 Fraction_lower_bound_186: Fraction[186] >= 0
 Fraction_upper_bound_186: Fraction[186] <= 1
 R1862: - 0.7428042507169217 alphas[0] + 0.0835726231279816 alphas[1]
   + 1.605253210030802 alphas[2] + 1.09660458177765 alphas[3] - alphas[4]
   + f_var[186] = 0
 R1863: - 2094.959652373807 Fraction[186] + Q0_var[186] = 0
 R1864: Q0_var[186] + Lost_0_expr_186 = 359.1198295807582
 R1865: Q0_var[186] + Q_hat_adjusted_186 = 470.9645283217769
 R1866: Sold_0[186] - Q0_var[186] - Q1_var[186]
   + total_stock_second_period_186 = 0
 R1867: total_stock_second_period_186 + Lost_1_expr_186
   = 111.8340838540373
 R1868: Sold_1[186] + Left_1[186] - total_stock_second_period_186 = 0
 R1869: - 700 Sold_0[186] - 700 Sold_1[186] + 700 Lost_0[186]
   + 700 Lost_1[186] + 300 Left_1[186] + profits_vars[186] = 0
 Fraction_lower_bound_187: Fraction[187] >= 0
 Fraction_upper_bound_187: Fraction[187] <= 1
 R1872: - 0.5095958601187965 alphas[0] + 0.4627868977124418 alphas[1]
   - 0.7691777100612548 alphas[2] - 1.582531062835244 alphas[3] - alphas[4]
   + f_var[187] = 0
 R1873: - 2094.959652373807 Fraction[187] + Q0_var[187] = 0
 R1874: Q0_var[187] + Lost_0_expr_187 = 1861.159359438815
 R1875: Q0_var[187] + Q_hat_adjusted_187 = 2310.640281712166
 R1876: Sold_0[187] - Q0_var[187] - Q1_var[187]
   + total_stock_second_period_187 = 0
 R1877: total_stock_second_period_187 + Lost_1_expr_187
   = 449.4703117095602
 R1878: Sold_1[187] + Left_1[187] - total_stock_second_period_187 = 0
 R1879: - 700 Sold_0[187] - 700 Sold_1[187] + 700 Lost_0[187]
   + 700 Lost_1[187] + 300 Left_1[187] + profits_vars[187] = 0
 Fraction_lower_bound_188: Fraction[188] >= 0
 Fraction_upper_bound_188: Fraction[188] <= 1
 R1882: - 1.614976935548151 alphas[0] - 0.996393863606167 alphas[1]
   - 1.662815331485259 alphas[2] + 0.542148419300902 alphas[3] - alphas[4]
   + f_var[188] = 0
 R1883: - 2094.959652373807 Fraction[188] + Q0_var[188] = 0
 R1884: Q0_var[188] + Lost_0_expr_188 = 2188.505075995322
 R1885: Q0_var[188] + Q_hat_adjusted_188 = 2765.058312629389
 R1886: Sold_0[188] - Q0_var[188] - Q1_var[188]
   + total_stock_second_period_188 = 0
 R1887: total_stock_second_period_188 + Lost_1_expr_188
   = 576.5426274432116
 R1888: Sold_1[188] + Left_1[188] - total_stock_second_period_188 = 0
 R1889: - 700 Sold_0[188] - 700 Sold_1[188] + 700 Lost_0[188]
   + 700 Lost_1[188] + 300 Left_1[188] + profits_vars[188] = 0
 Fraction_lower_bound_189: Fraction[189] >= 0
 Fraction_upper_bound_189: Fraction[189] <= 1
 R1892: 1.301676603083843 alphas[0] - 0.6354799849298 alphas[1]
   + 0.0823672708130883 alphas[2] - 0.0929764611196261 alphas[3]
   - alphas[4] + f_var[189] = 0
 R1893: - 2094.959652373807 Fraction[189] + Q0_var[189] = 0
 R1894: Q0_var[189] + Lost_0_expr_189 = 1499.796336489446
 R1895: Q0_var[189] + Q_hat_adjusted_189 = 1828.190378722097
 R1896: Sold_0[189] - Q0_var[189] - Q1_var[189]
   + total_stock_second_period_189 = 0
 R1897: total_stock_second_period_189 + Lost_1_expr_189
   = 328.3834303194924
 R1898: Sold_1[189] + Left_1[189] - total_stock_second_period_189 = 0
 R1899: - 700 Sold_0[189] - 700 Sold_1[189] + 700 Lost_0[189]
   + 700 Lost_1[189] + 300 Left_1[189] + profits_vars[189] = 0
 Fraction_lower_bound_190: Fraction[190] >= 0
 Fraction_upper_bound_190: Fraction[190] <= 1
 R1902: - 1.301532604895745 alphas[0] - 1.467478101085004 alphas[1]
   + 1.456787769816258 alphas[2] - 0.8306051674093151 alphas[3] - alphas[4]
   + f_var[190] = 0
 R1903: - 2094.959652373807 Fraction[190] + Q0_var[190] = 0
 R1904: Q0_var[190] + Lost_0_expr_190 = 593.9709921442616
 R1905: Q0_var[190] + Q_hat_adjusted_190 = 726.9269858976857
 R1906: Sold_0[190] - Q0_var[190] - Q1_var[190]
   + total_stock_second_period_190 = 0
 R1907: total_stock_second_period_190 + Lost_1_expr_190
   = 132.9453786759889
 R1908: Sold_1[190] + Left_1[190] - total_stock_second_period_190 = 0
 R1909: - 700 Sold_0[190] - 700 Sold_1[190] + 700 Lost_0[190]
   + 700 Lost_1[190] + 300 Left_1[190] + profits_vars[190] = 0
 Fraction_lower_bound_191: Fraction[191] >= 0
 Fraction_upper_bound_191: Fraction[191] <= 1
 R1912: 1.554230927618384 alphas[0] + 1.578594199139581 alphas[1]
   + 0.6944940249910981 alphas[2] - 1.784837550862486 alphas[3] - alphas[4]
   + f_var[191] = 0
 R1913: - 2094.959652373807 Fraction[191] + Q0_var[191] = 0
 R1914: Q0_var[191] + Lost_0_expr_191 = 1000.260275127421
 R1915: Q0_var[191] + Q_hat_adjusted_191 = 1241.611918391211
 R1916: Sold_0[191] - Q0_var[191] - Q1_var[191]
   + total_stock_second_period_191 = 0
 R1917: total_stock_second_period_191 + Lost_1_expr_191
   = 241.3410293382373
 R1918: Sold_1[191] + Left_1[191] - total_stock_second_period_191 = 0
 R1919: - 700 Sold_0[191] - 700 Sold_1[191] + 700 Lost_0[191]
   + 700 Lost_1[191] + 300 Left_1[191] + profits_vars[191] = 0
 Fraction_lower_bound_192: Fraction[192] >= 0
 Fraction_upper_bound_192: Fraction[192] <= 1
 R1922: - 0.169248081451756 alphas[0] - 0.7991675036655024 alphas[1]
   + 1.169780198042246 alphas[2] + 1.152467349479564 alphas[3] - alphas[4]
   + f_var[192] = 0
 R1923: - 2094.959652373807 Fraction[192] + Q0_var[192] = 0
 R1924: Q0_var[192] + Lost_0_expr_192 = 442.074196588126
 R1925: Q0_var[192] + Q_hat_adjusted_192 = 615.8417165782116
 R1926: Sold_0[192] - Q0_var[192] - Q1_var[192]
   + total_stock_second_period_192 = 0
 R1927: total_stock_second_period_192 + Lost_1_expr_192 = 173.756906649892
 R1928: Sold_1[192] + Left_1[192] - total_stock_second_period_192 = 0
 R1929: - 700 Sold_0[192] - 700 Sold_1[192] + 700 Lost_0[192]
   + 700 Lost_1[192] + 300 Left_1[192] + profits_vars[192] = 0
 Fraction_lower_bound_193: Fraction[193] >= 0
 Fraction_upper_bound_193: Fraction[193] <= 1
 R1932: 0.2760766692227348 alphas[0] + 1.466931875707243 alphas[1]
   - 0.072944753417579 alphas[2] - 1.013702117143944 alphas[3] - alphas[4]
   + f_var[193] = 0
 R1933: - 2094.959652373807 Fraction[193] + Q0_var[193] = 0
 R1934: Q0_var[193] + Lost_0_expr_193 = 1548.201342472243
 R1935: Q0_var[193] + Q_hat_adjusted_193 = 1898.680242250512
 R1936: Sold_0[193] - Q0_var[193] - Q1_var[193]
   + total_stock_second_period_193 = 0
 R1937: total_stock_second_period_193 + Lost_1_expr_193
   = 350.4682867614392
 R1938: Sold_1[193] + Left_1[193] - total_stock_second_period_193 = 0
 R1939: - 700 Sold_0[193] - 700 Sold_1[193] + 700 Lost_0[193]
   + 700 Lost_1[193] + 300 Left_1[193] + profits_vars[193] = 0
 Fraction_lower_bound_194: Fraction[194] >= 0
 Fraction_upper_bound_194: Fraction[194] <= 1
 R1942: - 0.1326708279550102 alphas[0] + 0.7048082839412101 alphas[1]
   + 1.052725318125413 alphas[2] + 0.8389309167794528 alphas[3] - alphas[4]
   + f_var[194] = 0
 R1943: - 2094.959652373807 Fraction[194] + Q0_var[194] = 0
 R1944: Q0_var[194] + Lost_0_expr_194 = 557.7831892918081
 R1945: Q0_var[194] + Q_hat_adjusted_194 = 748.1955259318718
 R1946: Sold_0[194] - Q0_var[194] - Q1_var[194]
   + total_stock_second_period_194 = 0
 R1947: total_stock_second_period_194 + Lost_1_expr_194
   = 190.4017236336499
 R1948: Sold_1[194] + Left_1[194] - total_stock_second_period_194 = 0
 R1949: - 700 Sold_0[194] - 700 Sold_1[194] + 700 Lost_0[194]
   + 700 Lost_1[194] + 300 Left_1[194] + profits_vars[194] = 0
 Fraction_lower_bound_195: Fraction[195] >= 0
 Fraction_upper_bound_195: Fraction[195] <= 1
 R1952: 0.4094756501413954 alphas[0] - 1.373481498257315 alphas[1]
   - 1.815056716187887 alphas[2] + 0.4167509531509589 alphas[3] - alphas[4]
   + f_var[195] = 0
 R1953: - 2094.959652373807 Fraction[195] + Q0_var[195] = 0
 R1954: Q0_var[195] + Lost_0_expr_195 = 2453.201322462989
 R1955: Q0_var[195] + Q_hat_adjusted_195 = 3051.402780186769
 R1956: Sold_0[195] - Q0_var[195] - Q1_var[195]
   + total_stock_second_period_195 = 0
 R1957: total_stock_second_period_195 + Lost_1_expr_195
   = 598.1908490946607
 R1958: Sold_1[195] + Left_1[195] - total_stock_second_period_195 = 0
 R1959: - 700 Sold_0[195] - 700 Sold_1[195] + 700 Lost_0[195]
   + 700 Lost_1[195] + 300 Left_1[195] + profits_vars[195] = 0
 Fraction_lower_bound_196: Fraction[196] >= 0
 Fraction_upper_bound_196: Fraction[196] <= 1
 R1962: 1.0042711081263 alphas[0] - 0.9029814853071944 alphas[1]
   + 0.2131757144981654 alphas[2] - 0.2086720310068235 alphas[3]
   - alphas[4] + f_var[196] = 0
 R1963: - 2094.959652373807 Fraction[196] + Q0_var[196] = 0
 R1964: Q0_var[196] + Lost_0_expr_196 = 1324.405805391361
 R1965: Q0_var[196] + Q_hat_adjusted_196 = 1634.199320078753
 R1966: Sold_0[196] - Q0_var[196] - Q1_var[196]
   + total_stock_second_period_196 = 0
 R1967: total_stock_second_period_196 + Lost_1_expr_196
   = 309.7829020314259
 R1968: Sold_1[196] + Left_1[196] - total_stock_second_period_196 = 0
 R1969: - 700 Sold_0[196] - 700 Sold_1[196] + 700 Lost_0[196]
   + 700 Lost_1[196] + 300 Left_1[196] + profits_vars[196] = 0
 Fraction_lower_bound_197: Fraction[197] >= 0
 Fraction_upper_bound_197: Fraction[197] <= 1
 R1972: 1.588539670717234 alphas[0] - 0.5267846263991915 alphas[1]
   + 0.3264857225081262 alphas[2] + 1.384424120037192 alphas[3] - alphas[4]
   + f_var[197] = 0
 R1973: - 2094.959652373807 Fraction[197] + Q0_var[197] = 0
 R1974: Q0_var[197] + Lost_0_expr_197 = 1166.596765455613
 R1975: Q0_var[197] + Q_hat_adjusted_197 = 1460.277973332767
 R1976: Sold_0[197] - Q0_var[197] - Q1_var[197]
   + total_stock_second_period_197 = 0
 R1977: total_stock_second_period_197 + Lost_1_expr_197
   = 293.6705934530557
 R1978: Sold_1[197] + Left_1[197] - total_stock_second_period_197 = 0
 R1979: - 700 Sold_0[197] - 700 Sold_1[197] + 700 Lost_0[197]
   + 700 Lost_1[197] + 300 Left_1[197] + profits_vars[197] = 0
 Fraction_lower_bound_198: Fraction[198] >= 0
 Fraction_upper_bound_198: Fraction[198] <= 1
 R1982: - 0.1121072423211419 alphas[0] + 0.3439436128097955 alphas[1]
   + 1.462859186440906 alphas[2] - 1.453999465989099 alphas[3] - alphas[4]
   + f_var[198] = 0
 R1983: - 2094.959652373807 Fraction[198] + Q0_var[198] = 0
 R1984: Q0_var[198] + Lost_0_expr_198 = 543.7053752641302
 R1985: Q0_var[198] + Q_hat_adjusted_198 = 675.7980350246504
 R1986: Sold_0[198] - Q0_var[198] - Q1_var[198]
   + total_stock_second_period_198 = 0
 R1987: total_stock_second_period_198 + Lost_1_expr_198
   = 132.0820433096775
 R1988: Sold_1[198] + Left_1[198] - total_stock_second_period_198 = 0
 R1989: - 700 Sold_0[198] - 700 Sold_1[198] + 700 Lost_0[198]
   + 700 Lost_1[198] + 300 Left_1[198] + profits_vars[198] = 0
 Fraction_lower_bound_199: Fraction[199] >= 0
 Fraction_upper_bound_199: Fraction[199] <= 1
 R1992: - 1.218412274194177 alphas[0] + 1.505264236935883 alphas[1]
   - 0.416319981247256 alphas[2] + 0.0640102146186848 alphas[3] - alphas[4]
   + f_var[199] = 0
 R1993: - 2094.959652373807 Fraction[199] + Q0_var[199] = 0
 R1994: Q0_var[199] + Lost_0_expr_199 = 1837.44902778305
 R1995: Q0_var[199] + Q_hat_adjusted_199 = 2236.754748983314
 R1996: Sold_0[199] - Q0_var[199] - Q1_var[199]
   + total_stock_second_period_199 = 0
 R1997: total_stock_second_period_199 + Lost_1_expr_199
   = 399.2951090883211
 R1998: Sold_1[199] + Left_1[199] - total_stock_second_period_199 = 0
 R1999: - 700 Sold_0[199] - 700 Sold_1[199] + 700 Lost_0[199]
   + 700 Lost_1[199] + 300 Left_1[199] + profits_vars[199] = 0
 Fraction_lower_bound_200: Fraction[200] >= 0
 Fraction_upper_bound_200: Fraction[200] <= 1
 R2002: 0.3818166527822575 alphas[0] + 0.8067992586102691 alphas[1]
   + 1.251394228367395 alphas[2] + 0.0655041428939916 alphas[3] - alphas[4]
   + f_var[200] = 0
 R2003: - 2094.959652373807 Fraction[200] + Q0_var[200] = 0
 R2004: Q0_var[200] + Lost_0_expr_200 = 965.1500352182386
 R2005: Q0_var[200] + Q_hat_adjusted_200 = 1127.312309749181
 R2006: Sold_0[200] - Q0_var[200] - Q1_var[200]
   + total_stock_second_period_200 = 0
 R2007: total_stock_second_period_200 + Lost_1_expr_200
   = 162.1516613225906
 R2008: Sold_1[200] + Left_1[200] - total_stock_second_period_200 = 0
 R2009: - 700 Sold_0[200] - 700 Sold_1[200] + 700 Lost_0[200]
   + 700 Lost_1[200] + 300 Left_1[200] + profits_vars[200] = 0
 Fraction_lower_bound_201: Fraction[201] >= 0
 Fraction_upper_bound_201: Fraction[201] <= 1
 R2012: 0.8942056452899162 alphas[0] + 0.074508081162271 alphas[1]
   + 0.7681265152145924 alphas[2] + 0.0718921856118588 alphas[3]
   - alphas[4] + f_var[201] = 0
 R2013: - 2094.959652373807 Fraction[201] + Q0_var[201] = 0
 R2014: Q0_var[201] + Lost_0_expr_201 = 624.7340081762145
 R2015: Q0_var[201] + Q_hat_adjusted_201 = 855.6153556727241
 R2016: Sold_0[201] - Q0_var[201] - Q1_var[201]
   + total_stock_second_period_201 = 0
 R2017: total_stock_second_period_201 + Lost_1_expr_201
   = 230.8707326300203
 R2018: Sold_1[201] + Left_1[201] - total_stock_second_period_201 = 0
 R2019: - 700 Sold_0[201] - 700 Sold_1[201] + 700 Lost_0[201]
   + 700 Lost_1[201] + 300 Left_1[201] + profits_vars[201] = 0
 Fraction_lower_bound_202: Fraction[202] >= 0
 Fraction_upper_bound_202: Fraction[202] <= 1
 R2022: 1.37926196811121 alphas[0] + 0.2673277447149971 alphas[1]
   + 0.6366263302277034 alphas[2] + 1.430892349840847 alphas[3] - alphas[4]
   + f_var[202] = 0
 R2023: - 2094.959652373807 Fraction[202] + Q0_var[202] = 0
 R2024: Q0_var[202] + Lost_0_expr_202 = 981.6673444654128
 R2025: Q0_var[202] + Q_hat_adjusted_202 = 1231.247583166875
 R2026: Sold_0[202] - Q0_var[202] - Q1_var[202]
   + total_stock_second_period_202 = 0
 R2027: total_stock_second_period_202 + Lost_1_expr_202
   = 249.5696242445653
 R2028: Sold_1[202] + Left_1[202] - total_stock_second_period_202 = 0
 R2029: - 700 Sold_0[202] - 700 Sold_1[202] + 700 Lost_0[202]
   + 700 Lost_1[202] + 300 Left_1[202] + profits_vars[202] = 0
 Fraction_lower_bound_203: Fraction[203] >= 0
 Fraction_upper_bound_203: Fraction[203] <= 1
 R2032: 1.362975762540062 alphas[0] + 0.1499767488687407 alphas[1]
   + 1.343475591273385 alphas[2] + 1.121078148508667 alphas[3] - alphas[4]
   + f_var[203] = 0
 R2033: - 2094.959652373807 Fraction[203] + Q0_var[203] = 0
 R2034: Q0_var[203] + Lost_0_expr_203 = 687.0287927576528
 R2035: Q0_var[203] + Q_hat_adjusted_203 = 836.0974032224291
 R2036: Sold_0[203] - Q0_var[203] - Q1_var[203]
   + total_stock_second_period_203 = 0
 R2037: total_stock_second_period_203 + Lost_1_expr_203
   = 149.0579959040795
 R2038: Sold_1[203] + Left_1[203] - total_stock_second_period_203 = 0
 R2039: - 700 Sold_0[203] - 700 Sold_1[203] + 700 Lost_0[203]
   + 700 Lost_1[203] + 300 Left_1[203] + profits_vars[203] = 0
 Fraction_lower_bound_204: Fraction[204] >= 0
 Fraction_upper_bound_204: Fraction[204] <= 1
 R2042: 0.8992063398575411 alphas[0] + 1.570045791455523 alphas[1]
   - 1.230064708477554 alphas[2] + 0.750896373214247 alphas[3] - alphas[4]
   + f_var[204] = 0
 R2043: - 2094.959652373807 Fraction[204] + Q0_var[204] = 0
 R2044: Q0_var[204] + Lost_0_expr_204 = 1773.78762113651
 R2045: Q0_var[204] + Q_hat_adjusted_204 = 2288.80515071371
 R2046: Sold_0[204] - Q0_var[204] - Q1_var[204]
   + total_stock_second_period_204 = 0
 R2047: total_stock_second_period_204 + Lost_1_expr_204
   = 515.0069193637528
 R2048: Sold_1[204] + Left_1[204] - total_stock_second_period_204 = 0
 R2049: - 700 Sold_0[204] - 700 Sold_1[204] + 700 Lost_0[204]
   + 700 Lost_1[204] + 300 Left_1[204] + profits_vars[204] = 0
 Fraction_lower_bound_205: Fraction[205] >= 0
 Fraction_upper_bound_205: Fraction[205] <= 1
 R2052: 1.312868553695165 alphas[0] - 0.4307072391828096 alphas[1]
   + 1.299828237879418 alphas[2] - 0.3429159582195218 alphas[3] - alphas[4]
   + f_var[205] = 0
 R2053: - 2094.959652373807 Fraction[205] + Q0_var[205] = 0
 R2054: Q0_var[205] + Lost_0_expr_205 = 857.2230358368699
 R2055: Q0_var[205] + Q_hat_adjusted_205 = 1012.498154261139
 R2056: Sold_0[205] - Q0_var[205] - Q1_var[205]
   + total_stock_second_period_205 = 0
 R2057: total_stock_second_period_205 + Lost_1_expr_205
   = 155.2645052753531
 R2058: Sold_1[205] + Left_1[205] - total_stock_second_period_205 = 0
 R2059: - 700 Sold_0[205] - 700 Sold_1[205] + 700 Lost_0[205]
   + 700 Lost_1[205] + 300 Left_1[205] + profits_vars[205] = 0
 Fraction_lower_bound_206: Fraction[206] >= 0
 Fraction_upper_bound_206: Fraction[206] <= 1
 R2062: 0.7504733567657628 alphas[0] + 0.1737659511008106 alphas[1]
   - 0.00989285754429734 alphas[2] - 1.787971005196873 alphas[3]
   - alphas[4] + f_var[206] = 0
 R2063: - 2094.959652373807 Fraction[206] + Q0_var[206] = 0
 R2064: Q0_var[206] + Lost_0_expr_206 = 1209.444825216027
 R2065: Q0_var[206] + Q_hat_adjusted_206 = 1550.957953408446
 R2066: Sold_0[206] - Q0_var[206] - Q1_var[206]
   + total_stock_second_period_206 = 0
 R2067: total_stock_second_period_206 + Lost_1_expr_206
   = 341.5025155943215
 R2068: Sold_1[206] + Left_1[206] - total_stock_second_period_206 = 0
 R2069: - 700 Sold_0[206] - 700 Sold_1[206] + 700 Lost_0[206]
   + 700 Lost_1[206] + 300 Left_1[206] + profits_vars[206] = 0
 Fraction_lower_bound_207: Fraction[207] >= 0
 Fraction_upper_bound_207: Fraction[207] <= 1
 R2072: 1.428276607914833 alphas[0] + 1.614930096433699 alphas[1]
   + 0.6720584788423839 alphas[2] + 0.4356081413901215 alphas[3]
   - alphas[4] + f_var[207] = 0
 R2073: - 2094.959652373807 Fraction[207] + Q0_var[207] = 0
 R2074: Q0_var[207] + Lost_0_expr_207 = 921.3868048682986
 R2075: Q0_var[207] + Q_hat_adjusted_207 = 1165.928708268028
 R2076: Sold_0[207] - Q0_var[207] - Q1_var[207]
   + total_stock_second_period_207 = 0
 R2077: total_stock_second_period_207 + Lost_1_expr_207
   = 244.5312898370325
 R2078: Sold_1[207] + Left_1[207] - total_stock_second_period_207 = 0
 R2079: - 700 Sold_0[207] - 700 Sold_1[207] + 700 Lost_0[207]
   + 700 Lost_1[207] + 300 Left_1[207] + profits_vars[207] = 0
 Fraction_lower_bound_208: Fraction[208] >= 0
 Fraction_upper_bound_208: Fraction[208] <= 1
 R2082: 1.430132547178157 alphas[0] - 1.44687582760102 alphas[1]
   + 0.8567252678450116 alphas[2] - 1.264788551645199 alphas[3] - alphas[4]
   + f_var[208] = 0
 R2083: - 2094.959652373807 Fraction[208] + Q0_var[208] = 0
 R2084: Q0_var[208] + Lost_0_expr_208 = 1051.578369570644
 R2085: Q0_var[208] + Q_hat_adjusted_208 = 1269.861265682267
 R2086: Sold_0[208] - Q0_var[208] - Q1_var[208]
   + total_stock_second_period_208 = 0
 R2087: total_stock_second_period_208 + Lost_1_expr_208
   = 218.2722828765191
 R2088: Sold_1[208] + Left_1[208] - total_stock_second_period_208 = 0
 R2089: - 700 Sold_0[208] - 700 Sold_1[208] + 700 Lost_0[208]
   + 700 Lost_1[208] + 300 Left_1[208] + profits_vars[208] = 0
 Fraction_lower_bound_209: Fraction[209] >= 0
 Fraction_upper_bound_209: Fraction[209] <= 1
 R2092: - 1.264416950348514 alphas[0] + 1.014429505729628 alphas[1]
   + 0.9033111357969937 alphas[2] + 1.687769175602179 alphas[3] - alphas[4]
   + f_var[209] = 0
 R2093: - 2094.959652373807 Fraction[209] + Q0_var[209] = 0
 R2094: Q0_var[209] + Lost_0_expr_209 = 771.8647396875381
 R2095: Q0_var[209] + Q_hat_adjusted_209 = 983.5232805981641
 R2096: Sold_0[209] - Q0_var[209] - Q1_var[209]
   + total_stock_second_period_209 = 0
 R2097: total_stock_second_period_209 + Lost_1_expr_209
   = 211.6479264531634
 R2098: Sold_1[209] + Left_1[209] - total_stock_second_period_209 = 0
 R2099: - 700 Sold_0[209] - 700 Sold_1[209] + 700 Lost_0[209]
   + 700 Lost_1[209] + 300 Left_1[209] + profits_vars[209] = 0
 Fraction_lower_bound_210: Fraction[210] >= 0
 Fraction_upper_bound_210: Fraction[210] <= 1
 R2102: 1.101215206972098 alphas[0] + 1.107737835745719 alphas[1]
   - 0.6669381763962025 alphas[2] - 0.9988224863324924 alphas[3]
   - alphas[4] + f_var[210] = 0
 R2103: - 2094.959652373807 Fraction[210] + Q0_var[210] = 0
 R2104: Q0_var[210] + Lost_0_expr_210 = 2119.948004618859
 R2105: Q0_var[210] + Q_hat_adjusted_210 = 2554.890803734098
 R2106: Sold_0[210] - Q0_var[210] - Q1_var[210]
   + total_stock_second_period_210 = 0
 R2107: total_stock_second_period_210 + Lost_1_expr_210
   = 434.9321879881747
 R2108: Sold_1[210] + Left_1[210] - total_stock_second_period_210 = 0
 R2109: - 700 Sold_0[210] - 700 Sold_1[210] + 700 Lost_0[210]
   + 700 Lost_1[210] + 300 Left_1[210] + profits_vars[210] = 0
 Fraction_lower_bound_211: Fraction[211] >= 0
 Fraction_upper_bound_211: Fraction[211] <= 1
 R2112: - 0.253647251721543 alphas[0] - 0.8357282730926348 alphas[1]
   - 1.102718802720068 alphas[2] + 0.1036933004553957 alphas[3] - alphas[4]
   + f_var[211] = 0
 R2113: - 2094.959652373807 Fraction[211] + Q0_var[211] = 0
 R2114: Q0_var[211] + Lost_0_expr_211 = 1723.497761392136
 R2115: Q0_var[211] + Q_hat_adjusted_211 = 2220.407125521867
 R2116: Sold_0[211] - Q0_var[211] - Q1_var[211]
   + total_stock_second_period_211 = 0
 R2117: total_stock_second_period_211 + Lost_1_expr_211
   = 496.8987525231301
 R2118: Sold_1[211] + Left_1[211] - total_stock_second_period_211 = 0
 R2119: - 700 Sold_0[211] - 700 Sold_1[211] + 700 Lost_0[211]
   + 700 Lost_1[211] + 300 Left_1[211] + profits_vars[211] = 0
 Fraction_lower_bound_212: Fraction[212] >= 0
 Fraction_upper_bound_212: Fraction[212] <= 1
 R2122: - 0.9821703361248982 alphas[0] + 0.5133386054420376 alphas[1]
   + 0.2506324421738335 alphas[2] - 0.6205061818435369 alphas[3]
   - alphas[4] + f_var[212] = 0
 R2123: - 2094.959652373807 Fraction[212] + Q0_var[212] = 0
 R2124: Q0_var[212] + Lost_0_expr_212 = 1030.501836563487
 R2125: Q0_var[212] + Q_hat_adjusted_212 = 1334.969128651879
 R2126: Sold_0[212] - Q0_var[212] - Q1_var[212]
   + total_stock_second_period_212 = 0
 R2127: total_stock_second_period_212 + Lost_1_expr_212
   = 304.4566791739243
 R2128: Sold_1[212] + Left_1[212] - total_stock_second_period_212 = 0
 R2129: - 700 Sold_0[212] - 700 Sold_1[212] + 700 Lost_0[212]
   + 700 Lost_1[212] + 300 Left_1[212] + profits_vars[212] = 0
 Fraction_lower_bound_213: Fraction[213] >= 0
 Fraction_upper_bound_213: Fraction[213] <= 1
 R2132: 0.0982989692512982 alphas[0] - 0.3592568573824841 alphas[1]
   + 0.5684494468679077 alphas[2] + 0.1740271789007342 alphas[3]
   - alphas[4] + f_var[213] = 0
 R2133: - 2094.959652373807 Fraction[213] + Q0_var[213] = 0
 R2134: Q0_var[213] + Lost_0_expr_213 = 1203.44144579062
 R2135: Q0_var[213] + Q_hat_adjusted_213 = 1462.716211875359
 R2136: Sold_0[213] - Q0_var[213] - Q1_var[213]
   + total_stock_second_period_213 = 0
 R2137: total_stock_second_period_213 + Lost_1_expr_213
   = 259.2641516909673
 R2138: Sold_1[213] + Left_1[213] - total_stock_second_period_213 = 0
 R2139: - 700 Sold_0[213] - 700 Sold_1[213] + 700 Lost_0[213]
   + 700 Lost_1[213] + 300 Left_1[213] + profits_vars[213] = 0
 Fraction_lower_bound_214: Fraction[214] >= 0
 Fraction_upper_bound_214: Fraction[214] <= 1
 R2142: 1.131039274052246 alphas[0] + 1.650853497263007 alphas[1]
   - 0.8820319082768271 alphas[2] + 0.4437036986062609 alphas[3]
   - alphas[4] + f_var[214] = 0
 R2143: - 2094.959652373807 Fraction[214] + Q0_var[214] = 0
 R2144: Q0_var[214] + Lost_0_expr_214 = 1818.825917399222
 R2145: Q0_var[214] + Q_hat_adjusted_214 = 2284.354333429905
 R2146: Sold_0[214] - Q0_var[214] - Q1_var[214]
   + total_stock_second_period_214 = 0
 R2147: total_stock_second_period_214 + Lost_1_expr_214
   = 465.5178056419056
 R2148: Sold_1[214] + Left_1[214] - total_stock_second_period_214 = 0
 R2149: - 700 Sold_0[214] - 700 Sold_1[214] + 700 Lost_0[214]
   + 700 Lost_1[214] + 300 Left_1[214] + profits_vars[214] = 0
 Fraction_lower_bound_215: Fraction[215] >= 0
 Fraction_upper_bound_215: Fraction[215] <= 1
 R2152: 0.973500168066767 alphas[0] + 0.2385975237029986 alphas[1]
   + 0.7420261685926249 alphas[2] - 1.443841392996423 alphas[3] - alphas[4]
   + f_var[215] = 0
 R2153: - 2094.959652373807 Fraction[215] + Q0_var[215] = 0
 R2154: Q0_var[215] + Lost_0_expr_215 = 1086.400116806772
 R2155: Q0_var[215] + Q_hat_adjusted_215 = 1320.992845711355
 R2156: Sold_0[215] - Q0_var[215] - Q1_var[215]
   + total_stock_second_period_215 = 0
 R2157: total_stock_second_period_215 + Lost_1_expr_215
   = 234.5821156416472
 R2158: Sold_1[215] + Left_1[215] - total_stock_second_period_215 = 0
 R2159: - 700 Sold_0[215] - 700 Sold_1[215] + 700 Lost_0[215]
   + 700 Lost_1[215] + 300 Left_1[215] + profits_vars[215] = 0
 Fraction_lower_bound_216: Fraction[216] >= 0
 Fraction_upper_bound_216: Fraction[216] <= 1
 R2162: 0.1781308646139686 alphas[0] + 1.588225335443206 alphas[1]
   - 1.372286369790452 alphas[2] - 0.38515404628664 alphas[3] - alphas[4]
   + f_var[216] = 0
 R2163: - 2094.959652373807 Fraction[216] + Q0_var[216] = 0
 R2164: Q0_var[216] + Lost_0_expr_216 = 2024.545705082481
 R2165: Q0_var[216] + Q_hat_adjusted_216 = 2559.786681917674
 R2166: Sold_0[216] - Q0_var[216] - Q1_var[216]
   + total_stock_second_period_216 = 0
 R2167: total_stock_second_period_216 + Lost_1_expr_216
   = 535.2303694706526
 R2168: Sold_1[216] + Left_1[216] - total_stock_second_period_216 = 0
 R2169: - 700 Sold_0[216] - 700 Sold_1[216] + 700 Lost_0[216]
   + 700 Lost_1[216] + 300 Left_1[216] + profits_vars[216] = 0
 Fraction_lower_bound_217: Fraction[217] >= 0
 Fraction_upper_bound_217: Fraction[217] <= 1
 R2172: - 0.1464737088304857 alphas[0] + 0.975057840817834 alphas[1]
   + 0.6551278590861985 alphas[2] + 1.357459170484646 alphas[3] - alphas[4]
   + f_var[217] = 0
 R2173: - 2094.959652373807 Fraction[217] + Q0_var[217] = 0
 R2174: Q0_var[217] + Lost_0_expr_217 = 1046.30312518055
 R2175: Q0_var[217] + Q_hat_adjusted_217 = 1293.252506440378
 R2176: Sold_0[217] - Q0_var[217] - Q1_var[217]
   + total_stock_second_period_217 = 0
 R2177: total_stock_second_period_217 + Lost_1_expr_217
   = 246.9387680000446
 R2178: Sold_1[217] + Left_1[217] - total_stock_second_period_217 = 0
 R2179: - 700 Sold_0[217] - 700 Sold_1[217] + 700 Lost_0[217]
   + 700 Lost_1[217] + 300 Left_1[217] + profits_vars[217] = 0
 Fraction_lower_bound_218: Fraction[218] >= 0
 Fraction_upper_bound_218: Fraction[218] <= 1
 R2182: 0.4628423605072734 alphas[0] + 1.024331014379906 alphas[1]
   + 1.508259408783174 alphas[2] + 1.535539917135484 alphas[3] - alphas[4]
   + f_var[218] = 0
 R2183: - 2094.959652373807 Fraction[218] + Q0_var[218] = 0
 R2184: Q0_var[218] + Lost_0_expr_218 = 637.651619978532
 R2185: Q0_var[218] + Q_hat_adjusted_218 = 763.2885179097868
 R2186: Sold_0[218] - Q0_var[218] - Q1_var[218]
   + total_stock_second_period_218 = 0
 R2187: total_stock_second_period_218 + Lost_1_expr_218
   = 125.6262817679998
 R2188: Sold_1[218] + Left_1[218] - total_stock_second_period_218 = 0
 R2189: - 700 Sold_0[218] - 700 Sold_1[218] + 700 Lost_0[218]
   + 700 Lost_1[218] + 300 Left_1[218] + profits_vars[218] = 0
 Fraction_lower_bound_219: Fraction[219] >= 0
 Fraction_upper_bound_219: Fraction[219] <= 1
 R2192: - 1.009515726233106 alphas[0] - 1.552142237902549 alphas[1]
   + 0.0346314030238547 alphas[2] + 1.317311778872444 alphas[3] - alphas[4]
   + f_var[219] = 0
 R2193: - 2094.959652373807 Fraction[219] + Q0_var[219] = 0
 R2194: Q0_var[219] + Lost_0_expr_219 = 1308.078874664296
 R2195: Q0_var[219] + Q_hat_adjusted_219 = 1643.260800147342
 R2196: Sold_0[219] - Q0_var[219] - Q1_var[219]
   + total_stock_second_period_219 = 0
 R2197: total_stock_second_period_219 + Lost_1_expr_219
   = 335.1713129221378
 R2198: Sold_1[219] + Left_1[219] - total_stock_second_period_219 = 0
 R2199: - 700 Sold_0[219] - 700 Sold_1[219] + 700 Lost_0[219]
   + 700 Lost_1[219] + 300 Left_1[219] + profits_vars[219] = 0
 Fraction_lower_bound_220: Fraction[220] >= 0
 Fraction_upper_bound_220: Fraction[220] <= 1
 R2202: - 0.9057172751023321 alphas[0] + 0.6668565500742494 alphas[1]
   + 0.6339324955994513 alphas[2] + 1.069553336438001 alphas[3] - alphas[4]
   + f_var[220] = 0
 R2203: - 2094.959652373807 Fraction[220] + Q0_var[220] = 0
 R2204: Q0_var[220] + Lost_0_expr_220 = 790.9416768346563
 R2205: Q0_var[220] + Q_hat_adjusted_220 = 1040.904968174515
 R2206: Sold_0[220] - Q0_var[220] - Q1_var[220]
   + total_stock_second_period_220 = 0
 R2207: total_stock_second_period_220 + Lost_1_expr_220
   = 249.9526786238897
 R2208: Sold_1[220] + Left_1[220] - total_stock_second_period_220 = 0
 R2209: - 700 Sold_0[220] - 700 Sold_1[220] + 700 Lost_0[220]
   + 700 Lost_1[220] + 300 Left_1[220] + profits_vars[220] = 0
 Fraction_lower_bound_221: Fraction[221] >= 0
 Fraction_upper_bound_221: Fraction[221] <= 1
 R2212: - 1.506160336140664 alphas[0] - 1.115071904602061 alphas[1]
   + 0.0500637699450191 alphas[2] + 1.016855134271714 alphas[3] - alphas[4]
   + f_var[221] = 0
 R2213: - 2094.959652373807 Fraction[221] + Q0_var[221] = 0
 R2214: Q0_var[221] + Lost_0_expr_221 = 1428.246646450981
 R2215: Q0_var[221] + Q_hat_adjusted_221 = 1761.234140421798
 R2216: Sold_0[221] - Q0_var[221] - Q1_var[221]
   + total_stock_second_period_221 = 0
 R2217: total_stock_second_period_221 + Lost_1_expr_221
   = 332.9768813596003
 R2218: Sold_1[221] + Left_1[221] - total_stock_second_period_221 = 0
 R2219: - 700 Sold_0[221] - 700 Sold_1[221] + 700 Lost_0[221]
   + 700 Lost_1[221] + 300 Left_1[221] + profits_vars[221] = 0
 Fraction_lower_bound_222: Fraction[222] >= 0
 Fraction_upper_bound_222: Fraction[222] <= 1
 R2222: 1.555040943363305 alphas[0] + 0.2740387473870235 alphas[1]
   + 0.661636760458891 alphas[2] - 1.651617684966593 alphas[3] - alphas[4]
   + f_var[222] = 0
 R2223: - 2094.959652373807 Fraction[222] + Q0_var[222] = 0
 R2224: Q0_var[222] + Lost_0_expr_222 = 640.7158418571609
 R2225: Q0_var[222] + Q_hat_adjusted_222 = 886.7396790601458
 R2226: Sold_0[222] - Q0_var[222] - Q1_var[222]
   + total_stock_second_period_222 = 0
 R2227: total_stock_second_period_222 + Lost_1_expr_222
   = 246.0132237408496
 R2228: Sold_1[222] + Left_1[222] - total_stock_second_period_222 = 0
 R2229: - 700 Sold_0[222] - 700 Sold_1[222] + 700 Lost_0[222]
   + 700 Lost_1[222] + 300 Left_1[222] + profits_vars[222] = 0
 Fraction_lower_bound_223: Fraction[223] >= 0
 Fraction_upper_bound_223: Fraction[223] <= 1
 R2232: - 1.398740742619085 alphas[0] - 1.191010711992268 alphas[1]
   - 1.782119702999448 alphas[2] + 1.275850681778787 alphas[3] - alphas[4]
   + f_var[223] = 0
 R2233: - 2094.959652373807 Fraction[223] + Q0_var[223] = 0
 R2234: Q0_var[223] + Lost_0_expr_223 = 2362.364538664486
 R2235: Q0_var[223] + Q_hat_adjusted_223 = 2955.882463849367
 R2236: Sold_0[223] - Q0_var[223] - Q1_var[223]
   + total_stock_second_period_223 = 0
 R2237: total_stock_second_period_223 + Lost_1_expr_223
   = 593.5073146967519
 R2238: Sold_1[223] + Left_1[223] - total_stock_second_period_223 = 0
 R2239: - 700 Sold_0[223] - 700 Sold_1[223] + 700 Lost_0[223]
   + 700 Lost_1[223] + 300 Left_1[223] + profits_vars[223] = 0
 Fraction_lower_bound_224: Fraction[224] >= 0
 Fraction_upper_bound_224: Fraction[224] <= 1
 R2242: 0.3158636267199662 alphas[0] - 0.5293340724941307 alphas[1]
   - 1.738241864687759 alphas[2] - 0.7903521373283851 alphas[3] - alphas[4]
   + f_var[224] = 0
 R2243: - 2094.959652373807 Fraction[224] + Q0_var[224] = 0
 R2244: Q0_var[224] + Lost_0_expr_224 = 1811.335975700625
 R2245: Q0_var[224] + Q_hat_adjusted_224 = 2398.614613384734
 R2246: Sold_0[224] - Q0_var[224] - Q1_var[224]
   + total_stock_second_period_224 = 0
 R2247: total_stock_second_period_224 + Lost_1_expr_224
   = 587.2680311326061
 R2248: Sold_1[224] + Left_1[224] - total_stock_second_period_224 = 0
 R2249: - 700 Sold_0[224] - 700 Sold_1[224] + 700 Lost_0[224]
   + 700 Lost_1[224] + 300 Left_1[224] + profits_vars[224] = 0
 Fraction_lower_bound_225: Fraction[225] >= 0
 Fraction_upper_bound_225: Fraction[225] <= 1
 R2252: - 1.339719564198373 alphas[0] + 1.647115021546036 alphas[1]
   - 0.4829354098555772 alphas[2] - 1.430744681840571 alphas[3] - alphas[4]
   + f_var[225] = 0
 R2253: - 2094.959652373807 Fraction[225] + Q0_var[225] = 0
 R2254: Q0_var[225] + Lost_0_expr_225 = 1716.781094566551
 R2255: Q0_var[225] + Q_hat_adjusted_225 = 2125.55930920941
 R2256: Sold_0[225] - Q0_var[225] - Q1_var[225]
   + total_stock_second_period_225 = 0
 R2257: total_stock_second_period_225 + Lost_1_expr_225
   = 408.7676028306506
 R2258: Sold_1[225] + Left_1[225] - total_stock_second_period_225 = 0
 R2259: - 700 Sold_0[225] - 700 Sold_1[225] + 700 Lost_0[225]
   + 700 Lost_1[225] + 300 Left_1[225] + profits_vars[225] = 0
 Fraction_lower_bound_226: Fraction[226] >= 0
 Fraction_upper_bound_226: Fraction[226] <= 1
 R2262: - 0.7004344571755108 alphas[0] + 1.300222067222895 alphas[1]
   - 1.499221030604546 alphas[2] + 0.2058731335348941 alphas[3] - alphas[4]
   + f_var[226] = 0
 R2263: - 2094.959652373807 Fraction[226] + Q0_var[226] = 0
 R2264: Q0_var[226] + Lost_0_expr_226 = 2313.697709550831
 R2265: Q0_var[226] + Q_hat_adjusted_226 = 2866.988380390392
 R2266: Sold_0[226] - Q0_var[226] - Q1_var[226]
   + total_stock_second_period_226 = 0
 R2267: total_stock_second_period_226 + Lost_1_expr_226
   = 553.2800586397474
 R2268: Sold_1[226] + Left_1[226] - total_stock_second_period_226 = 0
 R2269: - 700 Sold_0[226] - 700 Sold_1[226] + 700 Lost_0[226]
   + 700 Lost_1[226] + 300 Left_1[226] + profits_vars[226] = 0
 Fraction_lower_bound_227: Fraction[227] >= 0
 Fraction_upper_bound_227: Fraction[227] <= 1
 R2272: - 1.71110262385908 alphas[0] + 1.566613375831102 alphas[1]
   + 0.5881135380145612 alphas[2] - 0.9514381562091858 alphas[3]
   - alphas[4] + f_var[227] = 0
 R2273: - 2094.959652373807 Fraction[227] + Q0_var[227] = 0
 R2274: Q0_var[227] + Lost_0_expr_227 = 1057.088594689651
 R2275: Q0_var[227] + Q_hat_adjusted_227 = 1313.567191021763
 R2276: Sold_0[227] - Q0_var[227] - Q1_var[227]
   + total_stock_second_period_227 = 0
 R2277: total_stock_second_period_227 + Lost_1_expr_227
   = 256.4679829320679
 R2278: Sold_1[227] + Left_1[227] - total_stock_second_period_227 = 0
 R2279: - 700 Sold_0[227] - 700 Sold_1[227] + 700 Lost_0[227]
   + 700 Lost_1[227] + 300 Left_1[227] + profits_vars[227] = 0
 Fraction_lower_bound_228: Fraction[228] >= 0
 Fraction_upper_bound_228: Fraction[228] <= 1
 R2282: - 0.9338694817738744 alphas[0] + 1.370442467640214 alphas[1]
   - 1.081956675222353 alphas[2] - 1.138268304200631 alphas[3] - alphas[4]
   + f_var[228] = 0
 R2283: - 2094.959652373807 Fraction[228] + Q0_var[228] = 0
 R2284: Q0_var[228] + Lost_0_expr_228 = 1323.518782914005
 R2285: Q0_var[228] + Q_hat_adjusted_228 = 1817.475839884647
 R2286: Sold_0[228] - Q0_var[228] - Q1_var[228]
   + total_stock_second_period_228 = 0
 R2287: total_stock_second_period_228 + Lost_1_expr_228
   = 493.9464466248087
 R2288: Sold_1[228] + Left_1[228] - total_stock_second_period_228 = 0
 R2289: - 700 Sold_0[228] - 700 Sold_1[228] + 700 Lost_0[228]
   + 700 Lost_1[228] + 300 Left_1[228] + profits_vars[228] = 0
 Fraction_lower_bound_229: Fraction[229] >= 0
 Fraction_upper_bound_229: Fraction[229] <= 1
 R2292: 0.4113679047988248 alphas[0] + 0.9393679822978502 alphas[1]
   - 1.001644113685751 alphas[2] + 0.1350282760075618 alphas[3] - alphas[4]
   + f_var[229] = 0
 R2293: - 2094.959652373807 Fraction[229] + Q0_var[229] = 0
 R2294: Q0_var[229] + Lost_0_expr_229 = 2030.42417283989
 R2295: Q0_var[229] + Q_hat_adjusted_229 = 2512.961051158175
 R2296: Sold_0[229] - Q0_var[229] - Q1_var[229]
   + total_stock_second_period_229 = 0
 R2297: total_stock_second_period_229 + Lost_1_expr_229
   = 482.5262658576931
 R2298: Sold_1[229] + Left_1[229] - total_stock_second_period_229 = 0
 R2299: - 700 Sold_0[229] - 700 Sold_1[229] + 700 Lost_0[229]
   + 700 Lost_1[229] + 300 Left_1[229] + profits_vars[229] = 0
 Fraction_lower_bound_230: Fraction[230] >= 0
 Fraction_upper_bound_230: Fraction[230] <= 1
 R2302: - 0.0538772305820722 alphas[0] - 0.936929286256032 alphas[1]
   + 0.4390997091553653 alphas[2] - 0.9682141936534729 alphas[3]
   - alphas[4] + f_var[230] = 0
 R2303: - 2094.959652373807 Fraction[230] + Q0_var[230] = 0
 R2304: Q0_var[230] + Lost_0_expr_230 = 1203.495015483821
 R2305: Q0_var[230] + Q_hat_adjusted_230 = 1481.162885339238
 R2306: Sold_0[230] - Q0_var[230] - Q1_var[230]
   + total_stock_second_period_230 = 0
 R2307: total_stock_second_period_230 + Lost_1_expr_230
   = 277.6572568116756
 R2308: Sold_1[230] + Left_1[230] - total_stock_second_period_230 = 0
 R2309: - 700 Sold_0[230] - 700 Sold_1[230] + 700 Lost_0[230]
   + 700 Lost_1[230] + 300 Left_1[230] + profits_vars[230] = 0
 Fraction_lower_bound_231: Fraction[231] >= 0
 Fraction_upper_bound_231: Fraction[231] <= 1
 R2312: 0.3710026231746032 alphas[0] - 0.7646506834499104 alphas[1]
   - 0.8484028535892543 alphas[2] + 1.684272842477627 alphas[3] - alphas[4]
   + f_var[231] = 0
 R2313: - 2094.959652373807 Fraction[231] + Q0_var[231] = 0
 R2314: Q0_var[231] + Lost_0_expr_231 = 1789.280656526285
 R2315: Q0_var[231] + Q_hat_adjusted_231 = 2250.027133085217
 R2316: Sold_0[231] - Q0_var[231] - Q1_var[231]
   + total_stock_second_period_231 = 0
 R2317: total_stock_second_period_231 + Lost_1_expr_231
   = 460.7358652304511
 R2318: Sold_1[231] + Left_1[231] - total_stock_second_period_231 = 0
 R2319: - 700 Sold_0[231] - 700 Sold_1[231] + 700 Lost_0[231]
   + 700 Lost_1[231] + 300 Left_1[231] + profits_vars[231] = 0
 Fraction_lower_bound_232: Fraction[232] >= 0
 Fraction_upper_bound_232: Fraction[232] <= 1
 R2322: 0.4101164801279929 alphas[0] + 0.8583787179059594 alphas[1]
   + 1.359293696712682 alphas[2] + 1.166977161076853 alphas[3] - alphas[4]
   + f_var[232] = 0
 R2323: - 2094.959652373807 Fraction[232] + Q0_var[232] = 0
 R2324: Q0_var[232] + Lost_0_expr_232 = 450.4858931973076
 R2325: Q0_var[232] + Q_hat_adjusted_232 = 597.3052202043727
 R2326: Sold_0[232] - Q0_var[232] - Q1_var[232]
   + total_stock_second_period_232 = 0
 R2327: total_stock_second_period_232 + Lost_1_expr_232
   = 146.8087135993258
 R2328: Sold_1[232] + Left_1[232] - total_stock_second_period_232 = 0
 R2329: - 700 Sold_0[232] - 700 Sold_1[232] + 700 Lost_0[232]
   + 700 Lost_1[232] + 300 Left_1[232] + profits_vars[232] = 0
 Fraction_lower_bound_233: Fraction[233] >= 0
 Fraction_upper_bound_233: Fraction[233] <= 1
 R2332: 0.7645658098091774 alphas[0] + 0.3747252598855476 alphas[1]
   + 0.8706615258776458 alphas[2] - 0.4942439794376284 alphas[3]
   - alphas[4] + f_var[233] = 0
 R2333: - 2094.959652373807 Fraction[233] + Q0_var[233] = 0
 R2334: Q0_var[233] + Lost_0_expr_233 = 929.9619383113279
 R2335: Q0_var[233] + Q_hat_adjusted_233 = 1146.263145336098
 R2336: Sold_0[233] - Q0_var[233] - Q1_var[233]
   + total_stock_second_period_233 = 0
 R2337: total_stock_second_period_233 + Lost_1_expr_233 = 216.290593056255
 R2338: Sold_1[233] + Left_1[233] - total_stock_second_period_233 = 0
 R2339: - 700 Sold_0[233] - 700 Sold_1[233] + 700 Lost_0[233]
   + 700 Lost_1[233] + 300 Left_1[233] + profits_vars[233] = 0
 Fraction_lower_bound_234: Fraction[234] >= 0
 Fraction_upper_bound_234: Fraction[234] <= 1
 R2342: - 0.0365210799429582 alphas[0] - 1.182243123456936 alphas[1]
   + 0.9504081870767715 alphas[2] + 0.6880989927928994 alphas[3]
   - alphas[4] + f_var[234] = 0
 R2343: - 2094.959652373807 Fraction[234] + Q0_var[234] = 0
 R2344: Q0_var[234] + Lost_0_expr_234 = 790.4798372181082
 R2345: Q0_var[234] + Q_hat_adjusted_234 = 995.4413326865988
 R2346: Sold_0[234] - Q0_var[234] - Q1_var[234]
   + total_stock_second_period_234 = 0
 R2347: total_stock_second_period_234 + Lost_1_expr_234
   = 204.9508814464238
 R2348: Sold_1[234] + Left_1[234] - total_stock_second_period_234 = 0
 R2349: - 700 Sold_0[234] - 700 Sold_1[234] + 700 Lost_0[234]
   + 700 Lost_1[234] + 300 Left_1[234] + profits_vars[234] = 0
 Fraction_lower_bound_235: Fraction[235] >= 0
 Fraction_upper_bound_235: Fraction[235] <= 1
 R2352: - 0.6696103401517508 alphas[0] + 1.430601150150019 alphas[1]
   - 0.5118957607995188 alphas[2] + 0.721862066355213 alphas[3] - alphas[4]
   + f_var[235] = 0
 R2353: - 2094.959652373807 Fraction[235] + Q0_var[235] = 0
 R2354: Q0_var[235] + Lost_0_expr_235 = 1596.6195067706
 R2355: Q0_var[235] + Q_hat_adjusted_235 = 2009.515788669869
 R2356: Sold_0[235] - Q0_var[235] - Q1_var[235]
   + total_stock_second_period_235 = 0
 R2357: total_stock_second_period_235 + Lost_1_expr_235
   = 412.8856690027548
 R2358: Sold_1[235] + Left_1[235] - total_stock_second_period_235 = 0
 R2359: - 700 Sold_0[235] - 700 Sold_1[235] + 700 Lost_0[235]
   + 700 Lost_1[235] + 300 Left_1[235] + profits_vars[235] = 0
 Fraction_lower_bound_236: Fraction[236] >= 0
 Fraction_upper_bound_236: Fraction[236] <= 1
 R2362: 0.7085539117113155 alphas[0] - 0.2041371712781685 alphas[1]
   + 0.97902907099297 alphas[2] + 0.3482079082171963 alphas[3] - alphas[4]
   + f_var[236] = 0
 R2363: - 2094.959652373807 Fraction[236] + Q0_var[236] = 0
 R2364: Q0_var[236] + Lost_0_expr_236 = 974.7963502925447
 R2365: Q0_var[236] + Q_hat_adjusted_236 = 1175.688051937178
 R2366: Sold_0[236] - Q0_var[236] - Q1_var[236]
   + total_stock_second_period_236 = 0
 R2367: total_stock_second_period_236 + Lost_1_expr_236
   = 200.8810863635611
 R2368: Sold_1[236] + Left_1[236] - total_stock_second_period_236 = 0
 R2369: - 700 Sold_0[236] - 700 Sold_1[236] + 700 Lost_0[236]
   + 700 Lost_1[236] + 300 Left_1[236] + profits_vars[236] = 0
 Fraction_lower_bound_237: Fraction[237] >= 0
 Fraction_upper_bound_237: Fraction[237] <= 1
 R2372: - 0.1333386134271443 alphas[0] + 1.288844037317921 alphas[1]
   + 0.6914168034182194 alphas[2] + 0.5205854135770117 alphas[3]
   - alphas[4] + f_var[237] = 0
 R2373: - 2094.959652373807 Fraction[237] + Q0_var[237] = 0
 R2374: Q0_var[237] + Lost_0_expr_237 = 1044.087969164673
 R2375: Q0_var[237] + Q_hat_adjusted_237 = 1285.877183106676
 R2376: Sold_0[237] - Q0_var[237] - Q1_var[237]
   + total_stock_second_period_237 = 0
 R2377: total_stock_second_period_237 + Lost_1_expr_237
   = 241.7786000737533
 R2378: Sold_1[237] + Left_1[237] - total_stock_second_period_237 = 0
 R2379: - 700 Sold_0[237] - 700 Sold_1[237] + 700 Lost_0[237]
   + 700 Lost_1[237] + 300 Left_1[237] + profits_vars[237] = 0
 Fraction_lower_bound_238: Fraction[238] >= 0
 Fraction_upper_bound_238: Fraction[238] <= 1
 R2382: 1.253685712051108 alphas[0] - 1.628910946399497 alphas[1]
   + 0.7307911709076448 alphas[2] + 0.7347459807440788 alphas[3]
   - alphas[4] + f_var[238] = 0
 R2383: - 2094.959652373807 Fraction[238] + Q0_var[238] = 0
 R2384: Q0_var[238] + Lost_0_expr_238 = 950.4955868498531
 R2385: Q0_var[238] + Q_hat_adjusted_238 = 1186.685897000446
 R2386: Sold_0[238] - Q0_var[238] - Q1_var[238]
   + total_stock_second_period_238 = 0
 R2387: total_stock_second_period_238 + Lost_1_expr_238
   = 236.1796951734634
 R2388: Sold_1[238] + Left_1[238] - total_stock_second_period_238 = 0
 R2389: - 700 Sold_0[238] - 700 Sold_1[238] + 700 Lost_0[238]
   + 700 Lost_1[238] + 300 Left_1[238] + profits_vars[238] = 0
 Fraction_lower_bound_239: Fraction[239] >= 0
 Fraction_upper_bound_239: Fraction[239] <= 1
 R2392: 1.108968150022738 alphas[0] + 0.2046999524500363 alphas[1]
   - 1.628581888539602 alphas[2] + 0.2143553958239567 alphas[3] - alphas[4]
   + f_var[239] = 0
 R2393: - 2094.959652373807 Fraction[239] + Q0_var[239] = 0
 R2394: Q0_var[239] + Lost_0_expr_239 = 2116.922775514241
 R2395: Q0_var[239] + Q_hat_adjusted_239 = 2688.608131867085
 R2396: Sold_0[239] - Q0_var[239] - Q1_var[239]
   + total_stock_second_period_239 = 0
 R2397: total_stock_second_period_239 + Lost_1_expr_239
   = 571.6747450193398
 R2398: Sold_1[239] + Left_1[239] - total_stock_second_period_239 = 0
 R2399: - 700 Sold_0[239] - 700 Sold_1[239] + 700 Lost_0[239]
   + 700 Lost_1[239] + 300 Left_1[239] + profits_vars[239] = 0
 Fraction_lower_bound_240: Fraction[240] >= 0
 Fraction_upper_bound_240: Fraction[240] <= 1
 R2402: 1.494196490829342 alphas[0] - 1.367404909421568 alphas[1]
   - 1.644933337069894 alphas[2] + 0.0930423022043782 alphas[3] - alphas[4]
   + f_var[240] = 0
 R2403: - 2094.959652373807 Fraction[240] + Q0_var[240] = 0
 R2404: Q0_var[240] + Lost_0_expr_240 = 2566.210670256047
 R2405: Q0_var[240] + Q_hat_adjusted_240 = 3140.221146686921
 R2406: Sold_0[240] - Q0_var[240] - Q1_var[240]
   + total_stock_second_period_240 = 0
 R2407: total_stock_second_period_240 + Lost_1_expr_240 = 573.999866948554
 R2408: Sold_1[240] + Left_1[240] - total_stock_second_period_240 = 0
 R2409: - 700 Sold_0[240] - 700 Sold_1[240] + 700 Lost_0[240]
   + 700 Lost_1[240] + 300 Left_1[240] + profits_vars[240] = 0
 Fraction_lower_bound_241: Fraction[241] >= 0
 Fraction_upper_bound_241: Fraction[241] <= 1
 R2412: - 1.654472042727139 alphas[0] + 1.162861093687088 alphas[1]
   - 1.235805901020722 alphas[2] - 0.5374051193195721 alphas[3] - alphas[4]
   + f_var[241] = 0
 R2413: - 2094.959652373807 Fraction[241] + Q0_var[241] = 0
 R2414: Q0_var[241] + Lost_0_expr_241 = 1873.832010661456
 R2415: Q0_var[241] + Q_hat_adjusted_241 = 2389.665918979109
 R2416: Sold_0[241] - Q0_var[241] - Q1_var[241]
   + total_stock_second_period_241 = 0
 R2417: total_stock_second_period_241 + Lost_1_expr_241 = 515.823297965149
 R2418: Sold_1[241] + Left_1[241] - total_stock_second_period_241 = 0
 R2419: - 700 Sold_0[241] - 700 Sold_1[241] + 700 Lost_0[241]
   + 700 Lost_1[241] + 300 Left_1[241] + profits_vars[241] = 0
 Fraction_lower_bound_242: Fraction[242] >= 0
 Fraction_upper_bound_242: Fraction[242] <= 1
 R2422: 1.640553301215436 alphas[0] - 1.616468366445536 alphas[1]
   + 0.1008096772817929 alphas[2] - 0.320661598222137 alphas[3] - alphas[4]
   + f_var[242] = 0
 R2423: - 2094.959652373807 Fraction[242] + Q0_var[242] = 0
 R2424: Q0_var[242] + Lost_0_expr_242 = 1684.619917154917
 R2425: Q0_var[242] + Q_hat_adjusted_242 = 2010.391510288759
 R2426: Sold_0[242] - Q0_var[242] - Q1_var[242]
   + total_stock_second_period_242 = 0
 R2427: total_stock_second_period_242 + Lost_1_expr_242
   = 325.7609810833633
 R2428: Sold_1[242] + Left_1[242] - total_stock_second_period_242 = 0
 R2429: - 700 Sold_0[242] - 700 Sold_1[242] + 700 Lost_0[242]
   + 700 Lost_1[242] + 300 Left_1[242] + profits_vars[242] = 0
 Fraction_lower_bound_243: Fraction[243] >= 0
 Fraction_upper_bound_243: Fraction[243] <= 1
 R2432: 1.045121738713003 alphas[0] + 1.473809253327114 alphas[1]
   - 0.0347209174911751 alphas[2] + 1.162416898766361 alphas[3] - alphas[4]
   + f_var[243] = 0
 R2433: - 2094.959652373807 Fraction[243] + Q0_var[243] = 0
 R2434: Q0_var[243] + Lost_0_expr_243 = 1898.759035566563
 R2435: Q0_var[243] + Q_hat_adjusted_243 = 2243.802631671508
 R2436: Sold_0[243] - Q0_var[243] - Q1_var[243]
   + total_stock_second_period_243 = 0
 R2437: total_stock_second_period_243 + Lost_1_expr_243
   = 345.0329836464568
 R2438: Sold_1[243] + Left_1[243] - total_stock_second_period_243 = 0
 R2439: - 700 Sold_0[243] - 700 Sold_1[243] + 700 Lost_0[243]
   + 700 Lost_1[243] + 300 Left_1[243] + profits_vars[243] = 0
 Fraction_lower_bound_244: Fraction[244] >= 0
 Fraction_upper_bound_244: Fraction[244] <= 1
 R2442: - 0.4348955833940029 alphas[0] + 1.475075936647667 alphas[1]
   + 0.3713668323050653 alphas[2] - 1.043204761183069 alphas[3] - alphas[4]
   + f_var[244] = 0
 R2443: - 2094.959652373807 Fraction[244] + Q0_var[244] = 0
 R2444: Q0_var[244] + Lost_0_expr_244 = 1175.431465401162
 R2445: Q0_var[244] + Q_hat_adjusted_244 = 1462.73072693402
 R2446: Sold_0[244] - Q0_var[244] - Q1_var[244]
   + total_stock_second_period_244 = 0
 R2447: total_stock_second_period_244 + Lost_1_expr_244 = 287.288648000149
 R2448: Sold_1[244] + Left_1[244] - total_stock_second_period_244 = 0
 R2449: - 700 Sold_0[244] - 700 Sold_1[244] + 700 Lost_0[244]
   + 700 Lost_1[244] + 300 Left_1[244] + profits_vars[244] = 0
 Fraction_lower_bound_245: Fraction[245] >= 0
 Fraction_upper_bound_245: Fraction[245] <= 1
 R2452: 1.376407504848392 alphas[0] - 1.061915772247163 alphas[1]
   + 0.0680818587006775 alphas[2] - 1.702254764003104 alphas[3] - alphas[4]
   + f_var[245] = 0
 R2453: - 2094.959652373807 Fraction[245] + Q0_var[245] = 0
 R2454: Q0_var[245] + Lost_0_expr_245 = 1118.279925248586
 R2455: Q0_var[245] + Q_hat_adjusted_245 = 1448.705306456374
 R2456: Sold_0[245] - Q0_var[245] - Q1_var[245]
   + total_stock_second_period_245 = 0
 R2457: total_stock_second_period_245 + Lost_1_expr_245
   = 330.4147686996303
 R2458: Sold_1[245] + Left_1[245] - total_stock_second_period_245 = 0
 R2459: - 700 Sold_0[245] - 700 Sold_1[245] + 700 Lost_0[245]
   + 700 Lost_1[245] + 300 Left_1[245] + profits_vars[245] = 0
 Fraction_lower_bound_246: Fraction[246] >= 0
 Fraction_upper_bound_246: Fraction[246] <= 1
 R2462: - 1.351729122498534 alphas[0] - 0.9070872796800521 alphas[1]
   + 1.190912770509204 alphas[2] + 0.6571511775856659 alphas[3] - alphas[4]
   + f_var[246] = 0
 R2463: - 2094.959652373807 Fraction[246] + Q0_var[246] = 0
 R2464: Q0_var[246] + Lost_0_expr_246 = 672.0829463123032
 R2465: Q0_var[246] + Q_hat_adjusted_246 = 842.8454829550243
 R2466: Sold_0[246] - Q0_var[246] - Q1_var[246]
   + total_stock_second_period_246 = 0
 R2467: total_stock_second_period_246 + Lost_1_expr_246
   = 170.7519247014437
 R2468: Sold_1[246] + Left_1[246] - total_stock_second_period_246 = 0
 R2469: - 700 Sold_0[246] - 700 Sold_1[246] + 700 Lost_0[246]
   + 700 Lost_1[246] + 300 Left_1[246] + profits_vars[246] = 0
 Fraction_lower_bound_247: Fraction[247] >= 0
 Fraction_upper_bound_247: Fraction[247] <= 1
 R2472: - 0.7987033105062863 alphas[0] + 0.8476798345243466 alphas[1]
   + 0.735625769207242 alphas[2] - 1.385101521374662 alphas[3] - alphas[4]
   + f_var[247] = 0
 R2473: - 2094.959652373807 Fraction[247] + Q0_var[247] = 0
 R2474: Q0_var[247] + Lost_0_expr_247 = 1200.708040540061
 R2475: Q0_var[247] + Q_hat_adjusted_247 = 1436.210884497139
 R2476: Sold_0[247] - Q0_var[247] - Q1_var[247]
   + total_stock_second_period_247 = 0
 R2477: total_stock_second_period_247 + Lost_1_expr_247
   = 235.4922312769147
 R2478: Sold_1[247] + Left_1[247] - total_stock_second_period_247 = 0
 R2479: - 700 Sold_0[247] - 700 Sold_1[247] + 700 Lost_0[247]
   + 700 Lost_1[247] + 300 Left_1[247] + profits_vars[247] = 0
 Fraction_lower_bound_248: Fraction[248] >= 0
 Fraction_upper_bound_248: Fraction[248] <= 1
 R2482: - 1.639676628735881 alphas[0] + 0.851300756652872 alphas[1]
   + 1.037612944625008 alphas[2] - 0.5251296611935654 alphas[3] - alphas[4]
   + f_var[248] = 0
 R2483: - 2094.959652373807 Fraction[248] + Q0_var[248] = 0
 R2484: Q0_var[248] + Lost_0_expr_248 = 870.7986115906817
 R2485: Q0_var[248] + Q_hat_adjusted_248 = 1063.359879947493
 R2486: Sold_0[248] - Q0_var[248] - Q1_var[248]
   + total_stock_second_period_248 = 0
 R2487: total_stock_second_period_248 + Lost_1_expr_248
   = 192.5506531895341
 R2488: Sold_1[248] + Left_1[248] - total_stock_second_period_248 = 0
 R2489: - 700 Sold_0[248] - 700 Sold_1[248] + 700 Lost_0[248]
   + 700 Lost_1[248] + 300 Left_1[248] + profits_vars[248] = 0
 Fraction_lower_bound_249: Fraction[249] >= 0
 Fraction_upper_bound_249: Fraction[249] <= 1
 R2492: - 0.0762753976525781 alphas[0] + 0.1017719560953974 alphas[1]
   - 1.40901512953955 alphas[2] + 0.4058269391873228 alphas[3] - alphas[4]
   + f_var[249] = 0
 R2493: - 2094.959652373807 Fraction[249] + Q0_var[249] = 0
 R2494: Q0_var[249] + Lost_0_expr_249 = 2022.246963987618
 R2495: Q0_var[249] + Q_hat_adjusted_249 = 2562.710651772702
 R2496: Sold_0[249] - Q0_var[249] - Q1_var[249]
   + total_stock_second_period_249 = 0
 R2497: total_stock_second_period_249 + Lost_1_expr_249
   = 540.4530776953865
 R2498: Sold_1[249] + Left_1[249] - total_stock_second_period_249 = 0
 R2499: - 700 Sold_0[249] - 700 Sold_1[249] + 700 Lost_0[249]
   + 700 Lost_1[249] + 300 Left_1[249] + profits_vars[249] = 0
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
 f_var[150] free
 f_var[151] free
 f_var[152] free
 f_var[153] free
 f_var[154] free
 f_var[155] free
 f_var[156] free
 f_var[157] free
 f_var[158] free
 f_var[159] free
 f_var[160] free
 f_var[161] free
 f_var[162] free
 f_var[163] free
 f_var[164] free
 f_var[165] free
 f_var[166] free
 f_var[167] free
 f_var[168] free
 f_var[169] free
 f_var[170] free
 f_var[171] free
 f_var[172] free
 f_var[173] free
 f_var[174] free
 f_var[175] free
 f_var[176] free
 f_var[177] free
 f_var[178] free
 f_var[179] free
 f_var[180] free
 f_var[181] free
 f_var[182] free
 f_var[183] free
 f_var[184] free
 f_var[185] free
 f_var[186] free
 f_var[187] free
 f_var[188] free
 f_var[189] free
 f_var[190] free
 f_var[191] free
 f_var[192] free
 f_var[193] free
 f_var[194] free
 f_var[195] free
 f_var[196] free
 f_var[197] free
 f_var[198] free
 f_var[199] free
 f_var[200] free
 f_var[201] free
 f_var[202] free
 f_var[203] free
 f_var[204] free
 f_var[205] free
 f_var[206] free
 f_var[207] free
 f_var[208] free
 f_var[209] free
 f_var[210] free
 f_var[211] free
 f_var[212] free
 f_var[213] free
 f_var[214] free
 f_var[215] free
 f_var[216] free
 f_var[217] free
 f_var[218] free
 f_var[219] free
 f_var[220] free
 f_var[221] free
 f_var[222] free
 f_var[223] free
 f_var[224] free
 f_var[225] free
 f_var[226] free
 f_var[227] free
 f_var[228] free
 f_var[229] free
 f_var[230] free
 f_var[231] free
 f_var[232] free
 f_var[233] free
 f_var[234] free
 f_var[235] free
 f_var[236] free
 f_var[237] free
 f_var[238] free
 f_var[239] free
 f_var[240] free
 f_var[241] free
 f_var[242] free
 f_var[243] free
 f_var[244] free
 f_var[245] free
 f_var[246] free
 f_var[247] free
 f_var[248] free
 f_var[249] free
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
 Fraction[150] <= 1
 Fraction[151] <= 1
 Fraction[152] <= 1
 Fraction[153] <= 1
 Fraction[154] <= 1
 Fraction[155] <= 1
 Fraction[156] <= 1
 Fraction[157] <= 1
 Fraction[158] <= 1
 Fraction[159] <= 1
 Fraction[160] <= 1
 Fraction[161] <= 1
 Fraction[162] <= 1
 Fraction[163] <= 1
 Fraction[164] <= 1
 Fraction[165] <= 1
 Fraction[166] <= 1
 Fraction[167] <= 1
 Fraction[168] <= 1
 Fraction[169] <= 1
 Fraction[170] <= 1
 Fraction[171] <= 1
 Fraction[172] <= 1
 Fraction[173] <= 1
 Fraction[174] <= 1
 Fraction[175] <= 1
 Fraction[176] <= 1
 Fraction[177] <= 1
 Fraction[178] <= 1
 Fraction[179] <= 1
 Fraction[180] <= 1
 Fraction[181] <= 1
 Fraction[182] <= 1
 Fraction[183] <= 1
 Fraction[184] <= 1
 Fraction[185] <= 1
 Fraction[186] <= 1
 Fraction[187] <= 1
 Fraction[188] <= 1
 Fraction[189] <= 1
 Fraction[190] <= 1
 Fraction[191] <= 1
 Fraction[192] <= 1
 Fraction[193] <= 1
 Fraction[194] <= 1
 Fraction[195] <= 1
 Fraction[196] <= 1
 Fraction[197] <= 1
 Fraction[198] <= 1
 Fraction[199] <= 1
 Fraction[200] <= 1
 Fraction[201] <= 1
 Fraction[202] <= 1
 Fraction[203] <= 1
 Fraction[204] <= 1
 Fraction[205] <= 1
 Fraction[206] <= 1
 Fraction[207] <= 1
 Fraction[208] <= 1
 Fraction[209] <= 1
 Fraction[210] <= 1
 Fraction[211] <= 1
 Fraction[212] <= 1
 Fraction[213] <= 1
 Fraction[214] <= 1
 Fraction[215] <= 1
 Fraction[216] <= 1
 Fraction[217] <= 1
 Fraction[218] <= 1
 Fraction[219] <= 1
 Fraction[220] <= 1
 Fraction[221] <= 1
 Fraction[222] <= 1
 Fraction[223] <= 1
 Fraction[224] <= 1
 Fraction[225] <= 1
 Fraction[226] <= 1
 Fraction[227] <= 1
 Fraction[228] <= 1
 Fraction[229] <= 1
 Fraction[230] <= 1
 Fraction[231] <= 1
 Fraction[232] <= 1
 Fraction[233] <= 1
 Fraction[234] <= 1
 Fraction[235] <= 1
 Fraction[236] <= 1
 Fraction[237] <= 1
 Fraction[238] <= 1
 Fraction[239] <= 1
 Fraction[240] <= 1
 Fraction[241] <= 1
 Fraction[242] <= 1
 Fraction[243] <= 1
 Fraction[244] <= 1
 Fraction[245] <= 1
 Fraction[246] <= 1
 Fraction[247] <= 1
 Fraction[248] <= 1
 Fraction[249] <= 1
 Q0_var[0] <= 2095.959652373807
 Q0_var[1] <= 2095.959652373807
 Q0_var[2] <= 2095.959652373807
 Q0_var[3] <= 2095.959652373807
 Q0_var[4] <= 2095.959652373807
 Q0_var[5] <= 2095.959652373807
 Q0_var[6] <= 2095.959652373807
 Q0_var[7] <= 2095.959652373807
 Q0_var[8] <= 2095.959652373807
 Q0_var[9] <= 2095.959652373807
 Q0_var[10] <= 2095.959652373807
 Q0_var[11] <= 2095.959652373807
 Q0_var[12] <= 2095.959652373807
 Q0_var[13] <= 2095.959652373807
 Q0_var[14] <= 2095.959652373807
 Q0_var[15] <= 2095.959652373807
 Q0_var[16] <= 2095.959652373807
 Q0_var[17] <= 2095.959652373807
 Q0_var[18] <= 2095.959652373807
 Q0_var[19] <= 2095.959652373807
 Q0_var[20] <= 2095.959652373807
 Q0_var[21] <= 2095.959652373807
 Q0_var[22] <= 2095.959652373807
 Q0_var[23] <= 2095.959652373807
 Q0_var[24] <= 2095.959652373807
 Q0_var[25] <= 2095.959652373807
 Q0_var[26] <= 2095.959652373807
 Q0_var[27] <= 2095.959652373807
 Q0_var[28] <= 2095.959652373807
 Q0_var[29] <= 2095.959652373807
 Q0_var[30] <= 2095.959652373807
 Q0_var[31] <= 2095.959652373807
 Q0_var[32] <= 2095.959652373807
 Q0_var[33] <= 2095.959652373807
 Q0_var[34] <= 2095.959652373807
 Q0_var[35] <= 2095.959652373807
 Q0_var[36] <= 2095.959652373807
 Q0_var[37] <= 2095.959652373807
 Q0_var[38] <= 2095.959652373807
 Q0_var[39] <= 2095.959652373807
 Q0_var[40] <= 2095.959652373807
 Q0_var[41] <= 2095.959652373807
 Q0_var[42] <= 2095.959652373807
 Q0_var[43] <= 2095.959652373807
 Q0_var[44] <= 2095.959652373807
 Q0_var[45] <= 2095.959652373807
 Q0_var[46] <= 2095.959652373807
 Q0_var[47] <= 2095.959652373807
 Q0_var[48] <= 2095.959652373807
 Q0_var[49] <= 2095.959652373807
 Q0_var[50] <= 2095.959652373807
 Q0_var[51] <= 2095.959652373807
 Q0_var[52] <= 2095.959652373807
 Q0_var[53] <= 2095.959652373807
 Q0_var[54] <= 2095.959652373807
 Q0_var[55] <= 2095.959652373807
 Q0_var[56] <= 2095.959652373807
 Q0_var[57] <= 2095.959652373807
 Q0_var[58] <= 2095.959652373807
 Q0_var[59] <= 2095.959652373807
 Q0_var[60] <= 2095.959652373807
 Q0_var[61] <= 2095.959652373807
 Q0_var[62] <= 2095.959652373807
 Q0_var[63] <= 2095.959652373807
 Q0_var[64] <= 2095.959652373807
 Q0_var[65] <= 2095.959652373807
 Q0_var[66] <= 2095.959652373807
 Q0_var[67] <= 2095.959652373807
 Q0_var[68] <= 2095.959652373807
 Q0_var[69] <= 2095.959652373807
 Q0_var[70] <= 2095.959652373807
 Q0_var[71] <= 2095.959652373807
 Q0_var[72] <= 2095.959652373807
 Q0_var[73] <= 2095.959652373807
 Q0_var[74] <= 2095.959652373807
 Q0_var[75] <= 2095.959652373807
 Q0_var[76] <= 2095.959652373807
 Q0_var[77] <= 2095.959652373807
 Q0_var[78] <= 2095.959652373807
 Q0_var[79] <= 2095.959652373807
 Q0_var[80] <= 2095.959652373807
 Q0_var[81] <= 2095.959652373807
 Q0_var[82] <= 2095.959652373807
 Q0_var[83] <= 2095.959652373807
 Q0_var[84] <= 2095.959652373807
 Q0_var[85] <= 2095.959652373807
 Q0_var[86] <= 2095.959652373807
 Q0_var[87] <= 2095.959652373807
 Q0_var[88] <= 2095.959652373807
 Q0_var[89] <= 2095.959652373807
 Q0_var[90] <= 2095.959652373807
 Q0_var[91] <= 2095.959652373807
 Q0_var[92] <= 2095.959652373807
 Q0_var[93] <= 2095.959652373807
 Q0_var[94] <= 2095.959652373807
 Q0_var[95] <= 2095.959652373807
 Q0_var[96] <= 2095.959652373807
 Q0_var[97] <= 2095.959652373807
 Q0_var[98] <= 2095.959652373807
 Q0_var[99] <= 2095.959652373807
 Q0_var[100] <= 2095.959652373807
 Q0_var[101] <= 2095.959652373807
 Q0_var[102] <= 2095.959652373807
 Q0_var[103] <= 2095.959652373807
 Q0_var[104] <= 2095.959652373807
 Q0_var[105] <= 2095.959652373807
 Q0_var[106] <= 2095.959652373807
 Q0_var[107] <= 2095.959652373807
 Q0_var[108] <= 2095.959652373807
 Q0_var[109] <= 2095.959652373807
 Q0_var[110] <= 2095.959652373807
 Q0_var[111] <= 2095.959652373807
 Q0_var[112] <= 2095.959652373807
 Q0_var[113] <= 2095.959652373807
 Q0_var[114] <= 2095.959652373807
 Q0_var[115] <= 2095.959652373807
 Q0_var[116] <= 2095.959652373807
 Q0_var[117] <= 2095.959652373807
 Q0_var[118] <= 2095.959652373807
 Q0_var[119] <= 2095.959652373807
 Q0_var[120] <= 2095.959652373807
 Q0_var[121] <= 2095.959652373807
 Q0_var[122] <= 2095.959652373807
 Q0_var[123] <= 2095.959652373807
 Q0_var[124] <= 2095.959652373807
 Q0_var[125] <= 2095.959652373807
 Q0_var[126] <= 2095.959652373807
 Q0_var[127] <= 2095.959652373807
 Q0_var[128] <= 2095.959652373807
 Q0_var[129] <= 2095.959652373807
 Q0_var[130] <= 2095.959652373807
 Q0_var[131] <= 2095.959652373807
 Q0_var[132] <= 2095.959652373807
 Q0_var[133] <= 2095.959652373807
 Q0_var[134] <= 2095.959652373807
 Q0_var[135] <= 2095.959652373807
 Q0_var[136] <= 2095.959652373807
 Q0_var[137] <= 2095.959652373807
 Q0_var[138] <= 2095.959652373807
 Q0_var[139] <= 2095.959652373807
 Q0_var[140] <= 2095.959652373807
 Q0_var[141] <= 2095.959652373807
 Q0_var[142] <= 2095.959652373807
 Q0_var[143] <= 2095.959652373807
 Q0_var[144] <= 2095.959652373807
 Q0_var[145] <= 2095.959652373807
 Q0_var[146] <= 2095.959652373807
 Q0_var[147] <= 2095.959652373807
 Q0_var[148] <= 2095.959652373807
 Q0_var[149] <= 2095.959652373807
 Q0_var[150] <= 2095.959652373807
 Q0_var[151] <= 2095.959652373807
 Q0_var[152] <= 2095.959652373807
 Q0_var[153] <= 2095.959652373807
 Q0_var[154] <= 2095.959652373807
 Q0_var[155] <= 2095.959652373807
 Q0_var[156] <= 2095.959652373807
 Q0_var[157] <= 2095.959652373807
 Q0_var[158] <= 2095.959652373807
 Q0_var[159] <= 2095.959652373807
 Q0_var[160] <= 2095.959652373807
 Q0_var[161] <= 2095.959652373807
 Q0_var[162] <= 2095.959652373807
 Q0_var[163] <= 2095.959652373807
 Q0_var[164] <= 2095.959652373807
 Q0_var[165] <= 2095.959652373807
 Q0_var[166] <= 2095.959652373807
 Q0_var[167] <= 2095.959652373807
 Q0_var[168] <= 2095.959652373807
 Q0_var[169] <= 2095.959652373807
 Q0_var[170] <= 2095.959652373807
 Q0_var[171] <= 2095.959652373807
 Q0_var[172] <= 2095.959652373807
 Q0_var[173] <= 2095.959652373807
 Q0_var[174] <= 2095.959652373807
 Q0_var[175] <= 2095.959652373807
 Q0_var[176] <= 2095.959652373807
 Q0_var[177] <= 2095.959652373807
 Q0_var[178] <= 2095.959652373807
 Q0_var[179] <= 2095.959652373807
 Q0_var[180] <= 2095.959652373807
 Q0_var[181] <= 2095.959652373807
 Q0_var[182] <= 2095.959652373807
 Q0_var[183] <= 2095.959652373807
 Q0_var[184] <= 2095.959652373807
 Q0_var[185] <= 2095.959652373807
 Q0_var[186] <= 2095.959652373807
 Q0_var[187] <= 2095.959652373807
 Q0_var[188] <= 2095.959652373807
 Q0_var[189] <= 2095.959652373807
 Q0_var[190] <= 2095.959652373807
 Q0_var[191] <= 2095.959652373807
 Q0_var[192] <= 2095.959652373807
 Q0_var[193] <= 2095.959652373807
 Q0_var[194] <= 2095.959652373807
 Q0_var[195] <= 2095.959652373807
 Q0_var[196] <= 2095.959652373807
 Q0_var[197] <= 2095.959652373807
 Q0_var[198] <= 2095.959652373807
 Q0_var[199] <= 2095.959652373807
 Q0_var[200] <= 2095.959652373807
 Q0_var[201] <= 2095.959652373807
 Q0_var[202] <= 2095.959652373807
 Q0_var[203] <= 2095.959652373807
 Q0_var[204] <= 2095.959652373807
 Q0_var[205] <= 2095.959652373807
 Q0_var[206] <= 2095.959652373807
 Q0_var[207] <= 2095.959652373807
 Q0_var[208] <= 2095.959652373807
 Q0_var[209] <= 2095.959652373807
 Q0_var[210] <= 2095.959652373807
 Q0_var[211] <= 2095.959652373807
 Q0_var[212] <= 2095.959652373807
 Q0_var[213] <= 2095.959652373807
 Q0_var[214] <= 2095.959652373807
 Q0_var[215] <= 2095.959652373807
 Q0_var[216] <= 2095.959652373807
 Q0_var[217] <= 2095.959652373807
 Q0_var[218] <= 2095.959652373807
 Q0_var[219] <= 2095.959652373807
 Q0_var[220] <= 2095.959652373807
 Q0_var[221] <= 2095.959652373807
 Q0_var[222] <= 2095.959652373807
 Q0_var[223] <= 2095.959652373807
 Q0_var[224] <= 2095.959652373807
 Q0_var[225] <= 2095.959652373807
 Q0_var[226] <= 2095.959652373807
 Q0_var[227] <= 2095.959652373807
 Q0_var[228] <= 2095.959652373807
 Q0_var[229] <= 2095.959652373807
 Q0_var[230] <= 2095.959652373807
 Q0_var[231] <= 2095.959652373807
 Q0_var[232] <= 2095.959652373807
 Q0_var[233] <= 2095.959652373807
 Q0_var[234] <= 2095.959652373807
 Q0_var[235] <= 2095.959652373807
 Q0_var[236] <= 2095.959652373807
 Q0_var[237] <= 2095.959652373807
 Q0_var[238] <= 2095.959652373807
 Q0_var[239] <= 2095.959652373807
 Q0_var[240] <= 2095.959652373807
 Q0_var[241] <= 2095.959652373807
 Q0_var[242] <= 2095.959652373807
 Q0_var[243] <= 2095.959652373807
 Q0_var[244] <= 2095.959652373807
 Q0_var[245] <= 2095.959652373807
 Q0_var[246] <= 2095.959652373807
 Q0_var[247] <= 2095.959652373807
 Q0_var[248] <= 2095.959652373807
 Q0_var[249] <= 2095.959652373807
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
 profits_vars[150] free
 profits_vars[151] free
 profits_vars[152] free
 profits_vars[153] free
 profits_vars[154] free
 profits_vars[155] free
 profits_vars[156] free
 profits_vars[157] free
 profits_vars[158] free
 profits_vars[159] free
 profits_vars[160] free
 profits_vars[161] free
 profits_vars[162] free
 profits_vars[163] free
 profits_vars[164] free
 profits_vars[165] free
 profits_vars[166] free
 profits_vars[167] free
 profits_vars[168] free
 profits_vars[169] free
 profits_vars[170] free
 profits_vars[171] free
 profits_vars[172] free
 profits_vars[173] free
 profits_vars[174] free
 profits_vars[175] free
 profits_vars[176] free
 profits_vars[177] free
 profits_vars[178] free
 profits_vars[179] free
 profits_vars[180] free
 profits_vars[181] free
 profits_vars[182] free
 profits_vars[183] free
 profits_vars[184] free
 profits_vars[185] free
 profits_vars[186] free
 profits_vars[187] free
 profits_vars[188] free
 profits_vars[189] free
 profits_vars[190] free
 profits_vars[191] free
 profits_vars[192] free
 profits_vars[193] free
 profits_vars[194] free
 profits_vars[195] free
 profits_vars[196] free
 profits_vars[197] free
 profits_vars[198] free
 profits_vars[199] free
 profits_vars[200] free
 profits_vars[201] free
 profits_vars[202] free
 profits_vars[203] free
 profits_vars[204] free
 profits_vars[205] free
 profits_vars[206] free
 profits_vars[207] free
 profits_vars[208] free
 profits_vars[209] free
 profits_vars[210] free
 profits_vars[211] free
 profits_vars[212] free
 profits_vars[213] free
 profits_vars[214] free
 profits_vars[215] free
 profits_vars[216] free
 profits_vars[217] free
 profits_vars[218] free
 profits_vars[219] free
 profits_vars[220] free
 profits_vars[221] free
 profits_vars[222] free
 profits_vars[223] free
 profits_vars[224] free
 profits_vars[225] free
 profits_vars[226] free
 profits_vars[227] free
 profits_vars[228] free
 profits_vars[229] free
 profits_vars[230] free
 profits_vars[231] free
 profits_vars[232] free
 profits_vars[233] free
 profits_vars[234] free
 profits_vars[235] free
 profits_vars[236] free
 profits_vars[237] free
 profits_vars[238] free
 profits_vars[239] free
 profits_vars[240] free
 profits_vars[241] free
 profits_vars[242] free
 profits_vars[243] free
 profits_vars[244] free
 profits_vars[245] free
 profits_vars[246] free
 profits_vars[247] free
 profits_vars[248] free
 profits_vars[249] free
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
 Lost_0_expr_150 free
 Q_hat_adjusted_150 free
 Lost_1_expr_150 free
 Lost_0_expr_151 free
 Q_hat_adjusted_151 free
 Lost_1_expr_151 free
 Lost_0_expr_152 free
 Q_hat_adjusted_152 free
 Lost_1_expr_152 free
 Lost_0_expr_153 free
 Q_hat_adjusted_153 free
 Lost_1_expr_153 free
 Lost_0_expr_154 free
 Q_hat_adjusted_154 free
 Lost_1_expr_154 free
 Lost_0_expr_155 free
 Q_hat_adjusted_155 free
 Lost_1_expr_155 free
 Lost_0_expr_156 free
 Q_hat_adjusted_156 free
 Lost_1_expr_156 free
 Lost_0_expr_157 free
 Q_hat_adjusted_157 free
 Lost_1_expr_157 free
 Lost_0_expr_158 free
 Q_hat_adjusted_158 free
 Lost_1_expr_158 free
 Lost_0_expr_159 free
 Q_hat_adjusted_159 free
 Lost_1_expr_159 free
 Lost_0_expr_160 free
 Q_hat_adjusted_160 free
 Lost_1_expr_160 free
 Lost_0_expr_161 free
 Q_hat_adjusted_161 free
 Lost_1_expr_161 free
 Lost_0_expr_162 free
 Q_hat_adjusted_162 free
 Lost_1_expr_162 free
 Lost_0_expr_163 free
 Q_hat_adjusted_163 free
 Lost_1_expr_163 free
 Lost_0_expr_164 free
 Q_hat_adjusted_164 free
 Lost_1_expr_164 free
 Lost_0_expr_165 free
 Q_hat_adjusted_165 free
 Lost_1_expr_165 free
 Lost_0_expr_166 free
 Q_hat_adjusted_166 free
 Lost_1_expr_166 free
 Lost_0_expr_167 free
 Q_hat_adjusted_167 free
 Lost_1_expr_167 free
 Lost_0_expr_168 free
 Q_hat_adjusted_168 free
 Lost_1_expr_168 free
 Lost_0_expr_169 free
 Q_hat_adjusted_169 free
 Lost_1_expr_169 free
 Lost_0_expr_170 free
 Q_hat_adjusted_170 free
 Lost_1_expr_170 free
 Lost_0_expr_171 free
 Q_hat_adjusted_171 free
 Lost_1_expr_171 free
 Lost_0_expr_172 free
 Q_hat_adjusted_172 free
 Lost_1_expr_172 free
 Lost_0_expr_173 free
 Q_hat_adjusted_173 free
 Lost_1_expr_173 free
 Lost_0_expr_174 free
 Q_hat_adjusted_174 free
 Lost_1_expr_174 free
 Lost_0_expr_175 free
 Q_hat_adjusted_175 free
 Lost_1_expr_175 free
 Lost_0_expr_176 free
 Q_hat_adjusted_176 free
 Lost_1_expr_176 free
 Lost_0_expr_177 free
 Q_hat_adjusted_177 free
 Lost_1_expr_177 free
 Lost_0_expr_178 free
 Q_hat_adjusted_178 free
 Lost_1_expr_178 free
 Lost_0_expr_179 free
 Q_hat_adjusted_179 free
 Lost_1_expr_179 free
 Lost_0_expr_180 free
 Q_hat_adjusted_180 free
 Lost_1_expr_180 free
 Lost_0_expr_181 free
 Q_hat_adjusted_181 free
 Lost_1_expr_181 free
 Lost_0_expr_182 free
 Q_hat_adjusted_182 free
 Lost_1_expr_182 free
 Lost_0_expr_183 free
 Q_hat_adjusted_183 free
 Lost_1_expr_183 free
 Lost_0_expr_184 free
 Q_hat_adjusted_184 free
 Lost_1_expr_184 free
 Lost_0_expr_185 free
 Q_hat_adjusted_185 free
 Lost_1_expr_185 free
 Lost_0_expr_186 free
 Q_hat_adjusted_186 free
 Lost_1_expr_186 free
 Lost_0_expr_187 free
 Q_hat_adjusted_187 free
 Lost_1_expr_187 free
 Lost_0_expr_188 free
 Q_hat_adjusted_188 free
 Lost_1_expr_188 free
 Lost_0_expr_189 free
 Q_hat_adjusted_189 free
 Lost_1_expr_189 free
 Lost_0_expr_190 free
 Q_hat_adjusted_190 free
 Lost_1_expr_190 free
 Lost_0_expr_191 free
 Q_hat_adjusted_191 free
 Lost_1_expr_191 free
 Lost_0_expr_192 free
 Q_hat_adjusted_192 free
 Lost_1_expr_192 free
 Lost_0_expr_193 free
 Q_hat_adjusted_193 free
 Lost_1_expr_193 free
 Lost_0_expr_194 free
 Q_hat_adjusted_194 free
 Lost_1_expr_194 free
 Lost_0_expr_195 free
 Q_hat_adjusted_195 free
 Lost_1_expr_195 free
 Lost_0_expr_196 free
 Q_hat_adjusted_196 free
 Lost_1_expr_196 free
 Lost_0_expr_197 free
 Q_hat_adjusted_197 free
 Lost_1_expr_197 free
 Lost_0_expr_198 free
 Q_hat_adjusted_198 free
 Lost_1_expr_198 free
 Lost_0_expr_199 free
 Q_hat_adjusted_199 free
 Lost_1_expr_199 free
 Lost_0_expr_200 free
 Q_hat_adjusted_200 free
 Lost_1_expr_200 free
 Lost_0_expr_201 free
 Q_hat_adjusted_201 free
 Lost_1_expr_201 free
 Lost_0_expr_202 free
 Q_hat_adjusted_202 free
 Lost_1_expr_202 free
 Lost_0_expr_203 free
 Q_hat_adjusted_203 free
 Lost_1_expr_203 free
 Lost_0_expr_204 free
 Q_hat_adjusted_204 free
 Lost_1_expr_204 free
 Lost_0_expr_205 free
 Q_hat_adjusted_205 free
 Lost_1_expr_205 free
 Lost_0_expr_206 free
 Q_hat_adjusted_206 free
 Lost_1_expr_206 free
 Lost_0_expr_207 free
 Q_hat_adjusted_207 free
 Lost_1_expr_207 free
 Lost_0_expr_208 free
 Q_hat_adjusted_208 free
 Lost_1_expr_208 free
 Lost_0_expr_209 free
 Q_hat_adjusted_209 free
 Lost_1_expr_209 free
 Lost_0_expr_210 free
 Q_hat_adjusted_210 free
 Lost_1_expr_210 free
 Lost_0_expr_211 free
 Q_hat_adjusted_211 free
 Lost_1_expr_211 free
 Lost_0_expr_212 free
 Q_hat_adjusted_212 free
 Lost_1_expr_212 free
 Lost_0_expr_213 free
 Q_hat_adjusted_213 free
 Lost_1_expr_213 free
 Lost_0_expr_214 free
 Q_hat_adjusted_214 free
 Lost_1_expr_214 free
 Lost_0_expr_215 free
 Q_hat_adjusted_215 free
 Lost_1_expr_215 free
 Lost_0_expr_216 free
 Q_hat_adjusted_216 free
 Lost_1_expr_216 free
 Lost_0_expr_217 free
 Q_hat_adjusted_217 free
 Lost_1_expr_217 free
 Lost_0_expr_218 free
 Q_hat_adjusted_218 free
 Lost_1_expr_218 free
 Lost_0_expr_219 free
 Q_hat_adjusted_219 free
 Lost_1_expr_219 free
 Lost_0_expr_220 free
 Q_hat_adjusted_220 free
 Lost_1_expr_220 free
 Lost_0_expr_221 free
 Q_hat_adjusted_221 free
 Lost_1_expr_221 free
 Lost_0_expr_222 free
 Q_hat_adjusted_222 free
 Lost_1_expr_222 free
 Lost_0_expr_223 free
 Q_hat_adjusted_223 free
 Lost_1_expr_223 free
 Lost_0_expr_224 free
 Q_hat_adjusted_224 free
 Lost_1_expr_224 free
 Lost_0_expr_225 free
 Q_hat_adjusted_225 free
 Lost_1_expr_225 free
 Lost_0_expr_226 free
 Q_hat_adjusted_226 free
 Lost_1_expr_226 free
 Lost_0_expr_227 free
 Q_hat_adjusted_227 free
 Lost_1_expr_227 free
 Lost_0_expr_228 free
 Q_hat_adjusted_228 free
 Lost_1_expr_228 free
 Lost_0_expr_229 free
 Q_hat_adjusted_229 free
 Lost_1_expr_229 free
 Lost_0_expr_230 free
 Q_hat_adjusted_230 free
 Lost_1_expr_230 free
 Lost_0_expr_231 free
 Q_hat_adjusted_231 free
 Lost_1_expr_231 free
 Lost_0_expr_232 free
 Q_hat_adjusted_232 free
 Lost_1_expr_232 free
 Lost_0_expr_233 free
 Q_hat_adjusted_233 free
 Lost_1_expr_233 free
 Lost_0_expr_234 free
 Q_hat_adjusted_234 free
 Lost_1_expr_234 free
 Lost_0_expr_235 free
 Q_hat_adjusted_235 free
 Lost_1_expr_235 free
 Lost_0_expr_236 free
 Q_hat_adjusted_236 free
 Lost_1_expr_236 free
 Lost_0_expr_237 free
 Q_hat_adjusted_237 free
 Lost_1_expr_237 free
 Lost_0_expr_238 free
 Q_hat_adjusted_238 free
 Lost_1_expr_238 free
 Lost_0_expr_239 free
 Q_hat_adjusted_239 free
 Lost_1_expr_239 free
 Lost_0_expr_240 free
 Q_hat_adjusted_240 free
 Lost_1_expr_240 free
 Lost_0_expr_241 free
 Q_hat_adjusted_241 free
 Lost_1_expr_241 free
 Lost_0_expr_242 free
 Q_hat_adjusted_242 free
 Lost_1_expr_242 free
 Lost_0_expr_243 free
 Q_hat_adjusted_243 free
 Lost_1_expr_243 free
 Lost_0_expr_244 free
 Q_hat_adjusted_244 free
 Lost_1_expr_244 free
 Lost_0_expr_245 free
 Q_hat_adjusted_245 free
 Lost_1_expr_245 free
 Lost_0_expr_246 free
 Q_hat_adjusted_246 free
 Lost_1_expr_246 free
 Lost_0_expr_247 free
 Q_hat_adjusted_247 free
 Lost_1_expr_247 free
 Lost_0_expr_248 free
 Q_hat_adjusted_248 free
 Lost_1_expr_248 free
 Lost_0_expr_249 free
 Q_hat_adjusted_249 free
 Lost_1_expr_249 free
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 2209.591685417078 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 467.4681275293227 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 1460.924674130369 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 374.8527808345601 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1956.876720188578 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 398.759496512169 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 574.4938879510611 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 118.5106869860278 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 350.5583856119644 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 115.7278896878987 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1930.424972286104 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 506.0653019449017 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 1301.7713384587 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 245.5281836893697 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 2154.361153193902 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 503.6071292842974 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1788.712843767133 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 451.7520615648307 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 1470.725776623855 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 382.4738323575915 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 559.9662753110898 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 152.4804672376567 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 570.8335332569737 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 133.0212020421171 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 1220.288998083456 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 291.9907445769216 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 2208.330690747674 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 562.6259475861058 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 926.6836858879103 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 213.1316575590632 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 1333.071048766716 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 311.1203739596893 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1458.825881873058 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 420.8438978858483 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1518.131641077346 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 348.3509296391269 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 409.344249626055 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 112.5479770271992 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 2079.269866385042 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 447.6432523833461 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 2051.126239662301 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 559.2038372562537 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1341.583632819008 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 399.8471490916911 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 1378.224472273111 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 390.2063415823155 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 2391.879272655814 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 517.649120822622 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1971.702694486961 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 516.4192018164308 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 618.6502701757317 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 193.3782638824937 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 1381.884732282544 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 311.1495735734555 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 2235.997420321601 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 509.6348910084261 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 1812.715536432553 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 467.4169834297094 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 2084.481256144275 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 522.1940172850586 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 653.2976280459978 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 164.7216544587017 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 1033.924702446921 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 319.7335305048583 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 1591.035565029601 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 454.4766087298385 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 1894.230392865846 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 465.080627205861 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 680.50994738089 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 135.9024587626776 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 1910.45053084193 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 455.9701782549112 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 1644.206380798092 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 413.1142296021448 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 1114.189229344849 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 274.4565393660645 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 1273.167012356758 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 252.0542995868327 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 637.2819340078738 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 114.5046741567195 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 1102.71055919792 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 295.9949073194246 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 1009.456615620397 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 253.4975538191565 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 1041.928554297051 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 315.7506855840766 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 1505.032168234075 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 454.3728082212985 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 625.9527028651543 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 172.740953977432 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 1386.150798236841 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 339.9812159257486 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 1464.285830788816 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 496.4266441416203 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 1017.000072266852 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 224.3172953627362 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 2138.230680416611 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 531.8421927655245 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 2230.372189250569 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 566.4797319950662 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
 GC300: Fraction[50] = LOGISTIC ( f_var[50] )
 min_sales_constr_50: Sold_0[50] = MIN ( Q0_var[50] , 721.1527100297592 )
 max_lost_constr_50: Lost_0[50] = MAX ( Lost_0_expr_50 , 0 )
 max_Q1_constr_50: Q1_var[50] = MAX ( Q_hat_adjusted_50 , 0 )
 min_sales2_constr_50: Sold_1[50] = MIN ( total_stock_second_period_50
   , 156.4198045414643 )
 max_lost2_constr_50: Lost_1[50] = MAX ( Lost_1_expr_50 , 0 )
 GC306: Fraction[51] = LOGISTIC ( f_var[51] )
 min_sales_constr_51: Sold_0[51] = MIN ( Q0_var[51] , 514.7704737637375 )
 max_lost_constr_51: Lost_0[51] = MAX ( Lost_0_expr_51 , 0 )
 max_Q1_constr_51: Q1_var[51] = MAX ( Q_hat_adjusted_51 , 0 )
 min_sales2_constr_51: Sold_1[51] = MIN ( total_stock_second_period_51
   , 157.8502472190389 )
 max_lost2_constr_51: Lost_1[51] = MAX ( Lost_1_expr_51 , 0 )
 GC312: Fraction[52] = LOGISTIC ( f_var[52] )
 min_sales_constr_52: Sold_0[52] = MIN ( Q0_var[52] , 1747.116130576178 )
 max_lost_constr_52: Lost_0[52] = MAX ( Lost_0_expr_52 , 0 )
 max_Q1_constr_52: Q1_var[52] = MAX ( Q_hat_adjusted_52 , 0 )
 min_sales2_constr_52: Sold_1[52] = MIN ( total_stock_second_period_52
   , 422.6009981392555 )
 max_lost2_constr_52: Lost_1[52] = MAX ( Lost_1_expr_52 , 0 )
 GC318: Fraction[53] = LOGISTIC ( f_var[53] )
 min_sales_constr_53: Sold_0[53] = MIN ( Q0_var[53] , 1395.194890639617 )
 max_lost_constr_53: Lost_0[53] = MAX ( Lost_0_expr_53 , 0 )
 max_Q1_constr_53: Q1_var[53] = MAX ( Q_hat_adjusted_53 , 0 )
 min_sales2_constr_53: Sold_1[53] = MIN ( total_stock_second_period_53
   , 358.5949943345804 )
 max_lost2_constr_53: Lost_1[53] = MAX ( Lost_1_expr_53 , 0 )
 GC324: Fraction[54] = LOGISTIC ( f_var[54] )
 min_sales_constr_54: Sold_0[54] = MIN ( Q0_var[54] , 2121.712027700773 )
 max_lost_constr_54: Lost_0[54] = MAX ( Lost_0_expr_54 , 0 )
 max_Q1_constr_54: Q1_var[54] = MAX ( Q_hat_adjusted_54 , 0 )
 min_sales2_constr_54: Sold_1[54] = MIN ( total_stock_second_period_54
   , 551.7887870151146 )
 max_lost2_constr_54: Lost_1[54] = MAX ( Lost_1_expr_54 , 0 )
 GC330: Fraction[55] = LOGISTIC ( f_var[55] )
 min_sales_constr_55: Sold_0[55] = MIN ( Q0_var[55] , 1503.617297778341 )
 max_lost_constr_55: Lost_0[55] = MAX ( Lost_0_expr_55 , 0 )
 max_Q1_constr_55: Q1_var[55] = MAX ( Q_hat_adjusted_55 , 0 )
 min_sales2_constr_55: Sold_1[55] = MIN ( total_stock_second_period_55
   , 500.9759526288333 )
 max_lost2_constr_55: Lost_1[55] = MAX ( Lost_1_expr_55 , 0 )
 GC336: Fraction[56] = LOGISTIC ( f_var[56] )
 min_sales_constr_56: Sold_0[56] = MIN ( Q0_var[56] , 1528.813481598439 )
 max_lost_constr_56: Lost_0[56] = MAX ( Lost_0_expr_56 , 0 )
 max_Q1_constr_56: Q1_var[56] = MAX ( Q_hat_adjusted_56 , 0 )
 min_sales2_constr_56: Sold_1[56] = MIN ( total_stock_second_period_56
   , 350.9462809121204 )
 max_lost2_constr_56: Lost_1[56] = MAX ( Lost_1_expr_56 , 0 )
 GC342: Fraction[57] = LOGISTIC ( f_var[57] )
 min_sales_constr_57: Sold_0[57] = MIN ( Q0_var[57] , 558.006290834075 )
 max_lost_constr_57: Lost_0[57] = MAX ( Lost_0_expr_57 , 0 )
 max_Q1_constr_57: Q1_var[57] = MAX ( Q_hat_adjusted_57 , 0 )
 min_sales2_constr_57: Sold_1[57] = MIN ( total_stock_second_period_57
   , 134.3452543819624 )
 max_lost2_constr_57: Lost_1[57] = MAX ( Lost_1_expr_57 , 0 )
 GC348: Fraction[58] = LOGISTIC ( f_var[58] )
 min_sales_constr_58: Sold_0[58] = MIN ( Q0_var[58] , 862.2784349300385 )
 max_lost_constr_58: Lost_0[58] = MAX ( Lost_0_expr_58 , 0 )
 max_Q1_constr_58: Q1_var[58] = MAX ( Q_hat_adjusted_58 , 0 )
 min_sales2_constr_58: Sold_1[58] = MIN ( total_stock_second_period_58
   , 149.4352391997712 )
 max_lost2_constr_58: Lost_1[58] = MAX ( Lost_1_expr_58 , 0 )
 GC354: Fraction[59] = LOGISTIC ( f_var[59] )
 min_sales_constr_59: Sold_0[59] = MIN ( Q0_var[59] , 1582.803976715511 )
 max_lost_constr_59: Lost_0[59] = MAX ( Lost_0_expr_59 , 0 )
 max_Q1_constr_59: Q1_var[59] = MAX ( Q_hat_adjusted_59 , 0 )
 min_sales2_constr_59: Sold_1[59] = MIN ( total_stock_second_period_59
   , 324.2455054216166 )
 max_lost2_constr_59: Lost_1[59] = MAX ( Lost_1_expr_59 , 0 )
 GC360: Fraction[60] = LOGISTIC ( f_var[60] )
 min_sales_constr_60: Sold_0[60] = MIN ( Q0_var[60] , 810.8758598655588 )
 max_lost_constr_60: Lost_0[60] = MAX ( Lost_0_expr_60 , 0 )
 max_Q1_constr_60: Q1_var[60] = MAX ( Q_hat_adjusted_60 , 0 )
 min_sales2_constr_60: Sold_1[60] = MIN ( total_stock_second_period_60
   , 252.285809496679 )
 max_lost2_constr_60: Lost_1[60] = MAX ( Lost_1_expr_60 , 0 )
 GC366: Fraction[61] = LOGISTIC ( f_var[61] )
 min_sales_constr_61: Sold_0[61] = MIN ( Q0_var[61] , 2163.019327614861 )
 max_lost_constr_61: Lost_0[61] = MAX ( Lost_0_expr_61 , 0 )
 max_Q1_constr_61: Q1_var[61] = MAX ( Q_hat_adjusted_61 , 0 )
 min_sales2_constr_61: Sold_1[61] = MIN ( total_stock_second_period_61
   , 571.5647147866965 )
 max_lost2_constr_61: Lost_1[61] = MAX ( Lost_1_expr_61 , 0 )
 GC372: Fraction[62] = LOGISTIC ( f_var[62] )
 min_sales_constr_62: Sold_0[62] = MIN ( Q0_var[62] , 1645.146747751933 )
 max_lost_constr_62: Lost_0[62] = MAX ( Lost_0_expr_62 , 0 )
 max_Q1_constr_62: Q1_var[62] = MAX ( Q_hat_adjusted_62 , 0 )
 min_sales2_constr_62: Sold_1[62] = MIN ( total_stock_second_period_62
   , 432.9229111357785 )
 max_lost2_constr_62: Lost_1[62] = MAX ( Lost_1_expr_62 , 0 )
 GC378: Fraction[63] = LOGISTIC ( f_var[63] )
 min_sales_constr_63: Sold_0[63] = MIN ( Q0_var[63] , 916.2209094918736 )
 max_lost_constr_63: Lost_0[63] = MAX ( Lost_0_expr_63 , 0 )
 max_Q1_constr_63: Q1_var[63] = MAX ( Q_hat_adjusted_63 , 0 )
 min_sales2_constr_63: Sold_1[63] = MIN ( total_stock_second_period_63
   , 235.2652896022048 )
 max_lost2_constr_63: Lost_1[63] = MAX ( Lost_1_expr_63 , 0 )
 GC384: Fraction[64] = LOGISTIC ( f_var[64] )
 min_sales_constr_64: Sold_0[64] = MIN ( Q0_var[64] , 1117.461520177727 )
 max_lost_constr_64: Lost_0[64] = MAX ( Lost_0_expr_64 , 0 )
 max_Q1_constr_64: Q1_var[64] = MAX ( Q_hat_adjusted_64 , 0 )
 min_sales2_constr_64: Sold_1[64] = MIN ( total_stock_second_period_64
   , 278.4793111724015 )
 max_lost2_constr_64: Lost_1[64] = MAX ( Lost_1_expr_64 , 0 )
 GC390: Fraction[65] = LOGISTIC ( f_var[65] )
 min_sales_constr_65: Sold_0[65] = MIN ( Q0_var[65] , 587.4992360784864 )
 max_lost_constr_65: Lost_0[65] = MAX ( Lost_0_expr_65 , 0 )
 max_Q1_constr_65: Q1_var[65] = MAX ( Q_hat_adjusted_65 , 0 )
 min_sales2_constr_65: Sold_1[65] = MIN ( total_stock_second_period_65
   , 140.3913599461545 )
 max_lost2_constr_65: Lost_1[65] = MAX ( Lost_1_expr_65 , 0 )
 GC396: Fraction[66] = LOGISTIC ( f_var[66] )
 min_sales_constr_66: Sold_0[66] = MIN ( Q0_var[66] , 1075.330732676315 )
 max_lost_constr_66: Lost_0[66] = MAX ( Lost_0_expr_66 , 0 )
 max_Q1_constr_66: Q1_var[66] = MAX ( Q_hat_adjusted_66 , 0 )
 min_sales2_constr_66: Sold_1[66] = MIN ( total_stock_second_period_66
   , 224.8440222481231 )
 max_lost2_constr_66: Lost_1[66] = MAX ( Lost_1_expr_66 , 0 )
 GC402: Fraction[67] = LOGISTIC ( f_var[67] )
 min_sales_constr_67: Sold_0[67] = MIN ( Q0_var[67] , 1447.577073005654 )
 max_lost_constr_67: Lost_0[67] = MAX ( Lost_0_expr_67 , 0 )
 max_Q1_constr_67: Q1_var[67] = MAX ( Q_hat_adjusted_67 , 0 )
 min_sales2_constr_67: Sold_1[67] = MIN ( total_stock_second_period_67
   , 259.3857002051752 )
 max_lost2_constr_67: Lost_1[67] = MAX ( Lost_1_expr_67 , 0 )
 GC408: Fraction[68] = LOGISTIC ( f_var[68] )
 min_sales_constr_68: Sold_0[68] = MIN ( Q0_var[68] , 1464.106093317937 )
 max_lost_constr_68: Lost_0[68] = MAX ( Lost_0_expr_68 , 0 )
 max_Q1_constr_68: Q1_var[68] = MAX ( Q_hat_adjusted_68 , 0 )
 min_sales2_constr_68: Sold_1[68] = MIN ( total_stock_second_period_68
   , 420.9741722728962 )
 max_lost2_constr_68: Lost_1[68] = MAX ( Lost_1_expr_68 , 0 )
 GC414: Fraction[69] = LOGISTIC ( f_var[69] )
 min_sales_constr_69: Sold_0[69] = MIN ( Q0_var[69] , 1046.814392873435 )
 max_lost_constr_69: Lost_0[69] = MAX ( Lost_0_expr_69 , 0 )
 max_Q1_constr_69: Q1_var[69] = MAX ( Q_hat_adjusted_69 , 0 )
 min_sales2_constr_69: Sold_1[69] = MIN ( total_stock_second_period_69
   , 186.276867344065 )
 max_lost2_constr_69: Lost_1[69] = MAX ( Lost_1_expr_69 , 0 )
 GC420: Fraction[70] = LOGISTIC ( f_var[70] )
 min_sales_constr_70: Sold_0[70] = MIN ( Q0_var[70] , 1805.314202467221 )
 max_lost_constr_70: Lost_0[70] = MAX ( Lost_0_expr_70 , 0 )
 max_Q1_constr_70: Q1_var[70] = MAX ( Q_hat_adjusted_70 , 0 )
 min_sales2_constr_70: Sold_1[70] = MIN ( total_stock_second_period_70
   , 438.5918497130675 )
 max_lost2_constr_70: Lost_1[70] = MAX ( Lost_1_expr_70 , 0 )
 GC426: Fraction[71] = LOGISTIC ( f_var[71] )
 min_sales_constr_71: Sold_0[71] = MIN ( Q0_var[71] , 1027.159669445558 )
 max_lost_constr_71: Lost_0[71] = MAX ( Lost_0_expr_71 , 0 )
 max_Q1_constr_71: Q1_var[71] = MAX ( Q_hat_adjusted_71 , 0 )
 min_sales2_constr_71: Sold_1[71] = MIN ( total_stock_second_period_71
   , 158.8246778836932 )
 max_lost2_constr_71: Lost_1[71] = MAX ( Lost_1_expr_71 , 0 )
 GC432: Fraction[72] = LOGISTIC ( f_var[72] )
 min_sales_constr_72: Sold_0[72] = MIN ( Q0_var[72] , 2026.589469105 )
 max_lost_constr_72: Lost_0[72] = MAX ( Lost_0_expr_72 , 0 )
 max_Q1_constr_72: Q1_var[72] = MAX ( Q_hat_adjusted_72 , 0 )
 min_sales2_constr_72: Sold_1[72] = MIN ( total_stock_second_period_72
   , 547.0630553476674 )
 max_lost2_constr_72: Lost_1[72] = MAX ( Lost_1_expr_72 , 0 )
 GC438: Fraction[73] = LOGISTIC ( f_var[73] )
 min_sales_constr_73: Sold_0[73] = MIN ( Q0_var[73] , 1653.272965709546 )
 max_lost_constr_73: Lost_0[73] = MAX ( Lost_0_expr_73 , 0 )
 max_Q1_constr_73: Q1_var[73] = MAX ( Q_hat_adjusted_73 , 0 )
 min_sales2_constr_73: Sold_1[73] = MIN ( total_stock_second_period_73
   , 413.2182787607123 )
 max_lost2_constr_73: Lost_1[73] = MAX ( Lost_1_expr_73 , 0 )
 GC444: Fraction[74] = LOGISTIC ( f_var[74] )
 min_sales_constr_74: Sold_0[74] = MIN ( Q0_var[74] , 1507.986653427202 )
 max_lost_constr_74: Lost_0[74] = MAX ( Lost_0_expr_74 , 0 )
 max_Q1_constr_74: Q1_var[74] = MAX ( Q_hat_adjusted_74 , 0 )
 min_sales2_constr_74: Sold_1[74] = MIN ( total_stock_second_period_74
   , 388.7363100445972 )
 max_lost2_constr_74: Lost_1[74] = MAX ( Lost_1_expr_74 , 0 )
 GC450: Fraction[75] = LOGISTIC ( f_var[75] )
 min_sales_constr_75: Sold_0[75] = MIN ( Q0_var[75] , 854.2668527783042 )
 max_lost_constr_75: Lost_0[75] = MAX ( Lost_0_expr_75 , 0 )
 max_Q1_constr_75: Q1_var[75] = MAX ( Q_hat_adjusted_75 , 0 )
 min_sales2_constr_75: Sold_1[75] = MIN ( total_stock_second_period_75
   , 193.6358448848518 )
 max_lost2_constr_75: Lost_1[75] = MAX ( Lost_1_expr_75 , 0 )
 GC456: Fraction[76] = LOGISTIC ( f_var[76] )
 min_sales_constr_76: Sold_0[76] = MIN ( Q0_var[76] , 1802.793700748482 )
 max_lost_constr_76: Lost_0[76] = MAX ( Lost_0_expr_76 , 0 )
 max_Q1_constr_76: Q1_var[76] = MAX ( Q_hat_adjusted_76 , 0 )
 min_sales2_constr_76: Sold_1[76] = MIN ( total_stock_second_period_76
   , 470.4353253469187 )
 max_lost2_constr_76: Lost_1[76] = MAX ( Lost_1_expr_76 , 0 )
 GC462: Fraction[77] = LOGISTIC ( f_var[77] )
 min_sales_constr_77: Sold_0[77] = MIN ( Q0_var[77] , 886.701064888008 )
 max_lost_constr_77: Lost_0[77] = MAX ( Lost_0_expr_77 , 0 )
 max_Q1_constr_77: Q1_var[77] = MAX ( Q_hat_adjusted_77 , 0 )
 min_sales2_constr_77: Sold_1[77] = MIN ( total_stock_second_period_77
   , 218.0260809407732 )
 max_lost2_constr_77: Lost_1[77] = MAX ( Lost_1_expr_77 , 0 )
 GC468: Fraction[78] = LOGISTIC ( f_var[78] )
 min_sales_constr_78: Sold_0[78] = MIN ( Q0_var[78] , 2390.950436595269 )
 max_lost_constr_78: Lost_0[78] = MAX ( Lost_0_expr_78 , 0 )
 max_Q1_constr_78: Q1_var[78] = MAX ( Q_hat_adjusted_78 , 0 )
 min_sales2_constr_78: Sold_1[78] = MIN ( total_stock_second_period_78
   , 499.0216104899979 )
 max_lost2_constr_78: Lost_1[78] = MAX ( Lost_1_expr_78 , 0 )
 GC474: Fraction[79] = LOGISTIC ( f_var[79] )
 min_sales_constr_79: Sold_0[79] = MIN ( Q0_var[79] , 2161.675742814894 )
 max_lost_constr_79: Lost_0[79] = MAX ( Lost_0_expr_79 , 0 )
 max_Q1_constr_79: Q1_var[79] = MAX ( Q_hat_adjusted_79 , 0 )
 min_sales2_constr_79: Sold_1[79] = MIN ( total_stock_second_period_79
   , 489.4772935791971 )
 max_lost2_constr_79: Lost_1[79] = MAX ( Lost_1_expr_79 , 0 )
 GC480: Fraction[80] = LOGISTIC ( f_var[80] )
 min_sales_constr_80: Sold_0[80] = MIN ( Q0_var[80] , 2503.987807721447 )
 max_lost_constr_80: Lost_0[80] = MAX ( Lost_0_expr_80 , 0 )
 max_Q1_constr_80: Q1_var[80] = MAX ( Q_hat_adjusted_80 , 0 )
 min_sales2_constr_80: Sold_1[80] = MIN ( total_stock_second_period_80
   , 569.6978764065832 )
 max_lost2_constr_80: Lost_1[80] = MAX ( Lost_1_expr_80 , 0 )
 GC486: Fraction[81] = LOGISTIC ( f_var[81] )
 min_sales_constr_81: Sold_0[81] = MIN ( Q0_var[81] , 2277.635595132306 )
 max_lost_constr_81: Lost_0[81] = MAX ( Lost_0_expr_81 , 0 )
 max_Q1_constr_81: Q1_var[81] = MAX ( Q_hat_adjusted_81 , 0 )
 min_sales2_constr_81: Sold_1[81] = MIN ( total_stock_second_period_81
   , 491.3466157660876 )
 max_lost2_constr_81: Lost_1[81] = MAX ( Lost_1_expr_81 , 0 )
 GC492: Fraction[82] = LOGISTIC ( f_var[82] )
 min_sales_constr_82: Sold_0[82] = MIN ( Q0_var[82] , 1087.845412437135 )
 max_lost_constr_82: Lost_0[82] = MAX ( Lost_0_expr_82 , 0 )
 max_Q1_constr_82: Q1_var[82] = MAX ( Q_hat_adjusted_82 , 0 )
 min_sales2_constr_82: Sold_1[82] = MIN ( total_stock_second_period_82
   , 282.4725926701232 )
 max_lost2_constr_82: Lost_1[82] = MAX ( Lost_1_expr_82 , 0 )
 GC498: Fraction[83] = LOGISTIC ( f_var[83] )
 min_sales_constr_83: Sold_0[83] = MIN ( Q0_var[83] , 1543.614078096632 )
 max_lost_constr_83: Lost_0[83] = MAX ( Lost_0_expr_83 , 0 )
 max_Q1_constr_83: Q1_var[83] = MAX ( Q_hat_adjusted_83 , 0 )
 min_sales2_constr_83: Sold_1[83] = MIN ( total_stock_second_period_83
   , 355.326123189571 )
 max_lost2_constr_83: Lost_1[83] = MAX ( Lost_1_expr_83 , 0 )
 GC504: Fraction[84] = LOGISTIC ( f_var[84] )
 min_sales_constr_84: Sold_0[84] = MIN ( Q0_var[84] , 1996.724084129629 )
 max_lost_constr_84: Lost_0[84] = MAX ( Lost_0_expr_84 , 0 )
 max_Q1_constr_84: Q1_var[84] = MAX ( Q_hat_adjusted_84 , 0 )
 min_sales2_constr_84: Sold_1[84] = MIN ( total_stock_second_period_84
   , 413.4345205970489 )
 max_lost2_constr_84: Lost_1[84] = MAX ( Lost_1_expr_84 , 0 )
 GC510: Fraction[85] = LOGISTIC ( f_var[85] )
 min_sales_constr_85: Sold_0[85] = MIN ( Q0_var[85] , 947.280030676051 )
 max_lost_constr_85: Lost_0[85] = MAX ( Lost_0_expr_85 , 0 )
 max_Q1_constr_85: Q1_var[85] = MAX ( Q_hat_adjusted_85 , 0 )
 min_sales2_constr_85: Sold_1[85] = MIN ( total_stock_second_period_85
   , 233.759032516331 )
 max_lost2_constr_85: Lost_1[85] = MAX ( Lost_1_expr_85 , 0 )
 GC516: Fraction[86] = LOGISTIC ( f_var[86] )
 min_sales_constr_86: Sold_0[86] = MIN ( Q0_var[86] , 2065.909526868893 )
 max_lost_constr_86: Lost_0[86] = MAX ( Lost_0_expr_86 , 0 )
 max_Q1_constr_86: Q1_var[86] = MAX ( Q_hat_adjusted_86 , 0 )
 min_sales2_constr_86: Sold_1[86] = MIN ( total_stock_second_period_86
   , 500.0258091327837 )
 max_lost2_constr_86: Lost_1[86] = MAX ( Lost_1_expr_86 , 0 )
 GC522: Fraction[87] = LOGISTIC ( f_var[87] )
 min_sales_constr_87: Sold_0[87] = MIN ( Q0_var[87] , 964.8857025058114 )
 max_lost_constr_87: Lost_0[87] = MAX ( Lost_0_expr_87 , 0 )
 max_Q1_constr_87: Q1_var[87] = MAX ( Q_hat_adjusted_87 , 0 )
 min_sales2_constr_87: Sold_1[87] = MIN ( total_stock_second_period_87
   , 264.5490211721242 )
 max_lost2_constr_87: Lost_1[87] = MAX ( Lost_1_expr_87 , 0 )
 GC528: Fraction[88] = LOGISTIC ( f_var[88] )
 min_sales_constr_88: Sold_0[88] = MIN ( Q0_var[88] , 776.1424780634057 )
 max_lost_constr_88: Lost_0[88] = MAX ( Lost_0_expr_88 , 0 )
 max_Q1_constr_88: Q1_var[88] = MAX ( Q_hat_adjusted_88 , 0 )
 min_sales2_constr_88: Sold_1[88] = MIN ( total_stock_second_period_88
   , 165.0680390544581 )
 max_lost2_constr_88: Lost_1[88] = MAX ( Lost_1_expr_88 , 0 )
 GC534: Fraction[89] = LOGISTIC ( f_var[89] )
 min_sales_constr_89: Sold_0[89] = MIN ( Q0_var[89] , 421.9826301279803 )
 max_lost_constr_89: Lost_0[89] = MAX ( Lost_0_expr_89 , 0 )
 max_Q1_constr_89: Q1_var[89] = MAX ( Q_hat_adjusted_89 , 0 )
 min_sales2_constr_89: Sold_1[89] = MIN ( total_stock_second_period_89
   , 129.7607142723991 )
 max_lost2_constr_89: Lost_1[89] = MAX ( Lost_1_expr_89 , 0 )
 GC540: Fraction[90] = LOGISTIC ( f_var[90] )
 min_sales_constr_90: Sold_0[90] = MIN ( Q0_var[90] , 1723.416478483673 )
 max_lost_constr_90: Lost_0[90] = MAX ( Lost_0_expr_90 , 0 )
 max_Q1_constr_90: Q1_var[90] = MAX ( Q_hat_adjusted_90 , 0 )
 min_sales2_constr_90: Sold_1[90] = MIN ( total_stock_second_period_90
   , 430.0046904906585 )
 max_lost2_constr_90: Lost_1[90] = MAX ( Lost_1_expr_90 , 0 )
 GC546: Fraction[91] = LOGISTIC ( f_var[91] )
 min_sales_constr_91: Sold_0[91] = MIN ( Q0_var[91] , 540.6256937009452 )
 max_lost_constr_91: Lost_0[91] = MAX ( Lost_0_expr_91 , 0 )
 max_Q1_constr_91: Q1_var[91] = MAX ( Q_hat_adjusted_91 , 0 )
 min_sales2_constr_91: Sold_1[91] = MIN ( total_stock_second_period_91
   , 232.3556845884478 )
 max_lost2_constr_91: Lost_1[91] = MAX ( Lost_1_expr_91 , 0 )
 GC552: Fraction[92] = LOGISTIC ( f_var[92] )
 min_sales_constr_92: Sold_0[92] = MIN ( Q0_var[92] , 2256.509254840888 )
 max_lost_constr_92: Lost_0[92] = MAX ( Lost_0_expr_92 , 0 )
 max_Q1_constr_92: Q1_var[92] = MAX ( Q_hat_adjusted_92 , 0 )
 min_sales2_constr_92: Sold_1[92] = MIN ( total_stock_second_period_92
   , 525.7878492085347 )
 max_lost2_constr_92: Lost_1[92] = MAX ( Lost_1_expr_92 , 0 )
 GC558: Fraction[93] = LOGISTIC ( f_var[93] )
 min_sales_constr_93: Sold_0[93] = MIN ( Q0_var[93] , 2186.369116771933 )
 max_lost_constr_93: Lost_0[93] = MAX ( Lost_0_expr_93 , 0 )
 max_Q1_constr_93: Q1_var[93] = MAX ( Q_hat_adjusted_93 , 0 )
 min_sales2_constr_93: Sold_1[93] = MIN ( total_stock_second_period_93
   , 486.2159242828306 )
 max_lost2_constr_93: Lost_1[93] = MAX ( Lost_1_expr_93 , 0 )
 GC564: Fraction[94] = LOGISTIC ( f_var[94] )
 min_sales_constr_94: Sold_0[94] = MIN ( Q0_var[94] , 729.9747158876983 )
 max_lost_constr_94: Lost_0[94] = MAX ( Lost_0_expr_94 , 0 )
 max_Q1_constr_94: Q1_var[94] = MAX ( Q_hat_adjusted_94 , 0 )
 min_sales2_constr_94: Sold_1[94] = MIN ( total_stock_second_period_94
   , 157.6039757357379 )
 max_lost2_constr_94: Lost_1[94] = MAX ( Lost_1_expr_94 , 0 )
 GC570: Fraction[95] = LOGISTIC ( f_var[95] )
 min_sales_constr_95: Sold_0[95] = MIN ( Q0_var[95] , 1588.715952569364 )
 max_lost_constr_95: Lost_0[95] = MAX ( Lost_0_expr_95 , 0 )
 max_Q1_constr_95: Q1_var[95] = MAX ( Q_hat_adjusted_95 , 0 )
 min_sales2_constr_95: Sold_1[95] = MIN ( total_stock_second_period_95
   , 377.6406140004296 )
 max_lost2_constr_95: Lost_1[95] = MAX ( Lost_1_expr_95 , 0 )
 GC576: Fraction[96] = LOGISTIC ( f_var[96] )
 min_sales_constr_96: Sold_0[96] = MIN ( Q0_var[96] , 2584.39013815203 )
 max_lost_constr_96: Lost_0[96] = MAX ( Lost_0_expr_96 , 0 )
 max_Q1_constr_96: Q1_var[96] = MAX ( Q_hat_adjusted_96 , 0 )
 min_sales2_constr_96: Sold_1[96] = MIN ( total_stock_second_period_96
   , 583.6869480088644 )
 max_lost2_constr_96: Lost_1[96] = MAX ( Lost_1_expr_96 , 0 )
 GC582: Fraction[97] = LOGISTIC ( f_var[97] )
 min_sales_constr_97: Sold_0[97] = MIN ( Q0_var[97] , 2206.945975697695 )
 max_lost_constr_97: Lost_0[97] = MAX ( Lost_0_expr_97 , 0 )
 max_Q1_constr_97: Q1_var[97] = MAX ( Q_hat_adjusted_97 , 0 )
 min_sales2_constr_97: Sold_1[97] = MIN ( total_stock_second_period_97
   , 514.6154625736905 )
 max_lost2_constr_97: Lost_1[97] = MAX ( Lost_1_expr_97 , 0 )
 GC588: Fraction[98] = LOGISTIC ( f_var[98] )
 min_sales_constr_98: Sold_0[98] = MIN ( Q0_var[98] , 2382.086117672011 )
 max_lost_constr_98: Lost_0[98] = MAX ( Lost_0_expr_98 , 0 )
 max_Q1_constr_98: Q1_var[98] = MAX ( Q_hat_adjusted_98 , 0 )
 min_sales2_constr_98: Sold_1[98] = MIN ( total_stock_second_period_98
   , 536.1197821135552 )
 max_lost2_constr_98: Lost_1[98] = MAX ( Lost_1_expr_98 , 0 )
 GC594: Fraction[99] = LOGISTIC ( f_var[99] )
 min_sales_constr_99: Sold_0[99] = MIN ( Q0_var[99] , 696.2362741724085 )
 max_lost_constr_99: Lost_0[99] = MAX ( Lost_0_expr_99 , 0 )
 max_Q1_constr_99: Q1_var[99] = MAX ( Q_hat_adjusted_99 , 0 )
 min_sales2_constr_99: Sold_1[99] = MIN ( total_stock_second_period_99
   , 231.3783497707947 )
 max_lost2_constr_99: Lost_1[99] = MAX ( Lost_1_expr_99 , 0 )
 GC600: Fraction[100] = LOGISTIC ( f_var[100] )
 min_sales_constr_100: Sold_0[100] = MIN ( Q0_var[100] , 1871.557184261849 )
 max_lost_constr_100: Lost_0[100] = MAX ( Lost_0_expr_100 , 0 )
 max_Q1_constr_100: Q1_var[100] = MAX ( Q_hat_adjusted_100 , 0 )
 min_sales2_constr_100: Sold_1[100] = MIN ( total_stock_second_period_100
   , 462.7392045831747 )
 max_lost2_constr_100: Lost_1[100] = MAX ( Lost_1_expr_100 , 0 )
 GC606: Fraction[101] = LOGISTIC ( f_var[101] )
 min_sales_constr_101: Sold_0[101] = MIN ( Q0_var[101] , 992.2571077137886 )
 max_lost_constr_101: Lost_0[101] = MAX ( Lost_0_expr_101 , 0 )
 max_Q1_constr_101: Q1_var[101] = MAX ( Q_hat_adjusted_101 , 0 )
 min_sales2_constr_101: Sold_1[101] = MIN ( total_stock_second_period_101
   , 269.9408269761772 )
 max_lost2_constr_101: Lost_1[101] = MAX ( Lost_1_expr_101 , 0 )
 GC612: Fraction[102] = LOGISTIC ( f_var[102] )
 min_sales_constr_102: Sold_0[102] = MIN ( Q0_var[102] , 2134.903060812388 )
 max_lost_constr_102: Lost_0[102] = MAX ( Lost_0_expr_102 , 0 )
 max_Q1_constr_102: Q1_var[102] = MAX ( Q_hat_adjusted_102 , 0 )
 min_sales2_constr_102: Sold_1[102] = MIN ( total_stock_second_period_102
   , 514.0313313426161 )
 max_lost2_constr_102: Lost_1[102] = MAX ( Lost_1_expr_102 , 0 )
 GC618: Fraction[103] = LOGISTIC ( f_var[103] )
 min_sales_constr_103: Sold_0[103] = MIN ( Q0_var[103] , 265.183037246099 )
 max_lost_constr_103: Lost_0[103] = MAX ( Lost_0_expr_103 , 0 )
 max_Q1_constr_103: Q1_var[103] = MAX ( Q_hat_adjusted_103 , 0 )
 min_sales2_constr_103: Sold_1[103] = MIN ( total_stock_second_period_103
   , 151.3261088659059 )
 max_lost2_constr_103: Lost_1[103] = MAX ( Lost_1_expr_103 , 0 )
 GC624: Fraction[104] = LOGISTIC ( f_var[104] )
 min_sales_constr_104: Sold_0[104] = MIN ( Q0_var[104] , 1706.956062350333 )
 max_lost_constr_104: Lost_0[104] = MAX ( Lost_0_expr_104 , 0 )
 max_Q1_constr_104: Q1_var[104] = MAX ( Q_hat_adjusted_104 , 0 )
 min_sales2_constr_104: Sold_1[104] = MIN ( total_stock_second_period_104
   , 442.5570491507696 )
 max_lost2_constr_104: Lost_1[104] = MAX ( Lost_1_expr_104 , 0 )
 GC630: Fraction[105] = LOGISTIC ( f_var[105] )
 min_sales_constr_105: Sold_0[105] = MIN ( Q0_var[105] , 1584.403187941445 )
 max_lost_constr_105: Lost_0[105] = MAX ( Lost_0_expr_105 , 0 )
 max_Q1_constr_105: Q1_var[105] = MAX ( Q_hat_adjusted_105 , 0 )
 min_sales2_constr_105: Sold_1[105] = MIN ( total_stock_second_period_105
   , 422.4521598086443 )
 max_lost2_constr_105: Lost_1[105] = MAX ( Lost_1_expr_105 , 0 )
 GC636: Fraction[106] = LOGISTIC ( f_var[106] )
 min_sales_constr_106: Sold_0[106] = MIN ( Q0_var[106] , 1129.009875204798 )
 max_lost_constr_106: Lost_0[106] = MAX ( Lost_0_expr_106 , 0 )
 max_Q1_constr_106: Q1_var[106] = MAX ( Q_hat_adjusted_106 , 0 )
 min_sales2_constr_106: Sold_1[106] = MIN ( total_stock_second_period_106
   , 230.6590717572131 )
 max_lost2_constr_106: Lost_1[106] = MAX ( Lost_1_expr_106 , 0 )
 GC642: Fraction[107] = LOGISTIC ( f_var[107] )
 min_sales_constr_107: Sold_0[107] = MIN ( Q0_var[107] , 1489.323294919429 )
 max_lost_constr_107: Lost_0[107] = MAX ( Lost_0_expr_107 , 0 )
 max_Q1_constr_107: Q1_var[107] = MAX ( Q_hat_adjusted_107 , 0 )
 min_sales2_constr_107: Sold_1[107] = MIN ( total_stock_second_period_107
   , 416.0807377570629 )
 max_lost2_constr_107: Lost_1[107] = MAX ( Lost_1_expr_107 , 0 )
 GC648: Fraction[108] = LOGISTIC ( f_var[108] )
 min_sales_constr_108: Sold_0[108] = MIN ( Q0_var[108] , 2423.338290859486 )
 max_lost_constr_108: Lost_0[108] = MAX ( Lost_0_expr_108 , 0 )
 max_Q1_constr_108: Q1_var[108] = MAX ( Q_hat_adjusted_108 , 0 )
 min_sales2_constr_108: Sold_1[108] = MIN ( total_stock_second_period_108
   , 555.4404784956855 )
 max_lost2_constr_108: Lost_1[108] = MAX ( Lost_1_expr_108 , 0 )
 GC654: Fraction[109] = LOGISTIC ( f_var[109] )
 min_sales_constr_109: Sold_0[109] = MIN ( Q0_var[109] , 1347.512484001371 )
 max_lost_constr_109: Lost_0[109] = MAX ( Lost_0_expr_109 , 0 )
 max_Q1_constr_109: Q1_var[109] = MAX ( Q_hat_adjusted_109 , 0 )
 min_sales2_constr_109: Sold_1[109] = MIN ( total_stock_second_period_109
   , 407.5829149352639 )
 max_lost2_constr_109: Lost_1[109] = MAX ( Lost_1_expr_109 , 0 )
 GC660: Fraction[110] = LOGISTIC ( f_var[110] )
 min_sales_constr_110: Sold_0[110] = MIN ( Q0_var[110] , 808.6962681387486 )
 max_lost_constr_110: Lost_0[110] = MAX ( Lost_0_expr_110 , 0 )
 max_Q1_constr_110: Q1_var[110] = MAX ( Q_hat_adjusted_110 , 0 )
 min_sales2_constr_110: Sold_1[110] = MIN ( total_stock_second_period_110
   , 192.0215051734302 )
 max_lost2_constr_110: Lost_1[110] = MAX ( Lost_1_expr_110 , 0 )
 GC666: Fraction[111] = LOGISTIC ( f_var[111] )
 min_sales_constr_111: Sold_0[111] = MIN ( Q0_var[111] , 1081.933542856184 )
 max_lost_constr_111: Lost_0[111] = MAX ( Lost_0_expr_111 , 0 )
 max_Q1_constr_111: Q1_var[111] = MAX ( Q_hat_adjusted_111 , 0 )
 min_sales2_constr_111: Sold_1[111] = MIN ( total_stock_second_period_111
   , 253.0812067634787 )
 max_lost2_constr_111: Lost_1[111] = MAX ( Lost_1_expr_111 , 0 )
 GC672: Fraction[112] = LOGISTIC ( f_var[112] )
 min_sales_constr_112: Sold_0[112] = MIN ( Q0_var[112] , 1086.471625591272 )
 max_lost_constr_112: Lost_0[112] = MAX ( Lost_0_expr_112 , 0 )
 max_Q1_constr_112: Q1_var[112] = MAX ( Q_hat_adjusted_112 , 0 )
 min_sales2_constr_112: Sold_1[112] = MIN ( total_stock_second_period_112
   , 275.7792169063174 )
 max_lost2_constr_112: Lost_1[112] = MAX ( Lost_1_expr_112 , 0 )
 GC678: Fraction[113] = LOGISTIC ( f_var[113] )
 min_sales_constr_113: Sold_0[113] = MIN ( Q0_var[113] , 920.7942813195045 )
 max_lost_constr_113: Lost_0[113] = MAX ( Lost_0_expr_113 , 0 )
 max_Q1_constr_113: Q1_var[113] = MAX ( Q_hat_adjusted_113 , 0 )
 min_sales2_constr_113: Sold_1[113] = MIN ( total_stock_second_period_113
   , 255.5846742191091 )
 max_lost2_constr_113: Lost_1[113] = MAX ( Lost_1_expr_113 , 0 )
 GC684: Fraction[114] = LOGISTIC ( f_var[114] )
 min_sales_constr_114: Sold_0[114] = MIN ( Q0_var[114] , 1705.059422043449 )
 max_lost_constr_114: Lost_0[114] = MAX ( Lost_0_expr_114 , 0 )
 max_Q1_constr_114: Q1_var[114] = MAX ( Q_hat_adjusted_114 , 0 )
 min_sales2_constr_114: Sold_1[114] = MIN ( total_stock_second_period_114
   , 442.2037408344759 )
 max_lost2_constr_114: Lost_1[114] = MAX ( Lost_1_expr_114 , 0 )
 GC690: Fraction[115] = LOGISTIC ( f_var[115] )
 min_sales_constr_115: Sold_0[115] = MIN ( Q0_var[115] , 2231.806415904008 )
 max_lost_constr_115: Lost_0[115] = MAX ( Lost_0_expr_115 , 0 )
 max_Q1_constr_115: Q1_var[115] = MAX ( Q_hat_adjusted_115 , 0 )
 min_sales2_constr_115: Sold_1[115] = MIN ( total_stock_second_period_115
   , 575.0013901855997 )
 max_lost2_constr_115: Lost_1[115] = MAX ( Lost_1_expr_115 , 0 )
 GC696: Fraction[116] = LOGISTIC ( f_var[116] )
 min_sales_constr_116: Sold_0[116] = MIN ( Q0_var[116] , 1223.444593705643 )
 max_lost_constr_116: Lost_0[116] = MAX ( Lost_0_expr_116 , 0 )
 max_Q1_constr_116: Q1_var[116] = MAX ( Q_hat_adjusted_116 , 0 )
 min_sales2_constr_116: Sold_1[116] = MIN ( total_stock_second_period_116
   , 351.9550816663686 )
 max_lost2_constr_116: Lost_1[116] = MAX ( Lost_1_expr_116 , 0 )
 GC702: Fraction[117] = LOGISTIC ( f_var[117] )
 min_sales_constr_117: Sold_0[117] = MIN ( Q0_var[117] , 2015.064072540993 )
 max_lost_constr_117: Lost_0[117] = MAX ( Lost_0_expr_117 , 0 )
 max_Q1_constr_117: Q1_var[117] = MAX ( Q_hat_adjusted_117 , 0 )
 min_sales2_constr_117: Sold_1[117] = MIN ( total_stock_second_period_117
   , 495.1285260850203 )
 max_lost2_constr_117: Lost_1[117] = MAX ( Lost_1_expr_117 , 0 )
 GC708: Fraction[118] = LOGISTIC ( f_var[118] )
 min_sales_constr_118: Sold_0[118] = MIN ( Q0_var[118] , 2209.933624128001 )
 max_lost_constr_118: Lost_0[118] = MAX ( Lost_0_expr_118 , 0 )
 max_Q1_constr_118: Q1_var[118] = MAX ( Q_hat_adjusted_118 , 0 )
 min_sales2_constr_118: Sold_1[118] = MIN ( total_stock_second_period_118
   , 572.19253630176 )
 max_lost2_constr_118: Lost_1[118] = MAX ( Lost_1_expr_118 , 0 )
 GC714: Fraction[119] = LOGISTIC ( f_var[119] )
 min_sales_constr_119: Sold_0[119] = MIN ( Q0_var[119] , 1026.285556894711 )
 max_lost_constr_119: Lost_0[119] = MAX ( Lost_0_expr_119 , 0 )
 max_Q1_constr_119: Q1_var[119] = MAX ( Q_hat_adjusted_119 , 0 )
 min_sales2_constr_119: Sold_1[119] = MIN ( total_stock_second_period_119
   , 287.222207768883 )
 max_lost2_constr_119: Lost_1[119] = MAX ( Lost_1_expr_119 , 0 )
 GC720: Fraction[120] = LOGISTIC ( f_var[120] )
 min_sales_constr_120: Sold_0[120] = MIN ( Q0_var[120] , 312.9511939622993 )
 max_lost_constr_120: Lost_0[120] = MAX ( Lost_0_expr_120 , 0 )
 max_Q1_constr_120: Q1_var[120] = MAX ( Q_hat_adjusted_120 , 0 )
 min_sales2_constr_120: Sold_1[120] = MIN ( total_stock_second_period_120
   , 123.1393488977077 )
 max_lost2_constr_120: Lost_1[120] = MAX ( Lost_1_expr_120 , 0 )
 GC726: Fraction[121] = LOGISTIC ( f_var[121] )
 min_sales_constr_121: Sold_0[121] = MIN ( Q0_var[121] , 1965.264986901952 )
 max_lost_constr_121: Lost_0[121] = MAX ( Lost_0_expr_121 , 0 )
 max_Q1_constr_121: Q1_var[121] = MAX ( Q_hat_adjusted_121 , 0 )
 min_sales2_constr_121: Sold_1[121] = MIN ( total_stock_second_period_121
   , 476.6508023046482 )
 max_lost2_constr_121: Lost_1[121] = MAX ( Lost_1_expr_121 , 0 )
 GC732: Fraction[122] = LOGISTIC ( f_var[122] )
 min_sales_constr_122: Sold_0[122] = MIN ( Q0_var[122] , 2079.300835290032 )
 max_lost_constr_122: Lost_0[122] = MAX ( Lost_0_expr_122 , 0 )
 max_Q1_constr_122: Q1_var[122] = MAX ( Q_hat_adjusted_122 , 0 )
 min_sales2_constr_122: Sold_1[122] = MIN ( total_stock_second_period_122
   , 467.4951642809676 )
 max_lost2_constr_122: Lost_1[122] = MAX ( Lost_1_expr_122 , 0 )
 GC738: Fraction[123] = LOGISTIC ( f_var[123] )
 min_sales_constr_123: Sold_0[123] = MIN ( Q0_var[123] , 1074.558854238668 )
 max_lost_constr_123: Lost_0[123] = MAX ( Lost_0_expr_123 , 0 )
 max_Q1_constr_123: Q1_var[123] = MAX ( Q_hat_adjusted_123 , 0 )
 min_sales2_constr_123: Sold_1[123] = MIN ( total_stock_second_period_123
   , 251.6257888608569 )
 max_lost2_constr_123: Lost_1[123] = MAX ( Lost_1_expr_123 , 0 )
 GC744: Fraction[124] = LOGISTIC ( f_var[124] )
 min_sales_constr_124: Sold_0[124] = MIN ( Q0_var[124] , 735.7964543736533 )
 max_lost_constr_124: Lost_0[124] = MAX ( Lost_0_expr_124 , 0 )
 max_Q1_constr_124: Q1_var[124] = MAX ( Q_hat_adjusted_124 , 0 )
 min_sales2_constr_124: Sold_1[124] = MIN ( total_stock_second_period_124
   , 234.5649419146513 )
 max_lost2_constr_124: Lost_1[124] = MAX ( Lost_1_expr_124 , 0 )
 GC750: Fraction[125] = LOGISTIC ( f_var[125] )
 min_sales_constr_125: Sold_0[125] = MIN ( Q0_var[125] , 1458.777435761754 )
 max_lost_constr_125: Lost_0[125] = MAX ( Lost_0_expr_125 , 0 )
 max_Q1_constr_125: Q1_var[125] = MAX ( Q_hat_adjusted_125 , 0 )
 min_sales2_constr_125: Sold_1[125] = MIN ( total_stock_second_period_125
   , 451.6640786255443 )
 max_lost2_constr_125: Lost_1[125] = MAX ( Lost_1_expr_125 , 0 )
 GC756: Fraction[126] = LOGISTIC ( f_var[126] )
 min_sales_constr_126: Sold_0[126] = MIN ( Q0_var[126] , 592.1874092484778 )
 max_lost_constr_126: Lost_0[126] = MAX ( Lost_0_expr_126 , 0 )
 max_Q1_constr_126: Q1_var[126] = MAX ( Q_hat_adjusted_126 , 0 )
 min_sales2_constr_126: Sold_1[126] = MIN ( total_stock_second_period_126
   , 178.2025661024041 )
 max_lost2_constr_126: Lost_1[126] = MAX ( Lost_1_expr_126 , 0 )
 GC762: Fraction[127] = LOGISTIC ( f_var[127] )
 min_sales_constr_127: Sold_0[127] = MIN ( Q0_var[127] , 2051.28196134345 )
 max_lost_constr_127: Lost_0[127] = MAX ( Lost_0_expr_127 , 0 )
 max_Q1_constr_127: Q1_var[127] = MAX ( Q_hat_adjusted_127 , 0 )
 min_sales2_constr_127: Sold_1[127] = MIN ( total_stock_second_period_127
   , 521.7241126103153 )
 max_lost2_constr_127: Lost_1[127] = MAX ( Lost_1_expr_127 , 0 )
 GC768: Fraction[128] = LOGISTIC ( f_var[128] )
 min_sales_constr_128: Sold_0[128] = MIN ( Q0_var[128] , 2263.234217520333 )
 max_lost_constr_128: Lost_0[128] = MAX ( Lost_0_expr_128 , 0 )
 max_Q1_constr_128: Q1_var[128] = MAX ( Q_hat_adjusted_128 , 0 )
 min_sales2_constr_128: Sold_1[128] = MIN ( total_stock_second_period_128
   , 530.3145143476308 )
 max_lost2_constr_128: Lost_1[128] = MAX ( Lost_1_expr_128 , 0 )
 GC774: Fraction[129] = LOGISTIC ( f_var[129] )
 min_sales_constr_129: Sold_0[129] = MIN ( Q0_var[129] , 1837.421329608244 )
 max_lost_constr_129: Lost_0[129] = MAX ( Lost_0_expr_129 , 0 )
 max_Q1_constr_129: Q1_var[129] = MAX ( Q_hat_adjusted_129 , 0 )
 min_sales2_constr_129: Sold_1[129] = MIN ( total_stock_second_period_129
   , 383.1742144328632 )
 max_lost2_constr_129: Lost_1[129] = MAX ( Lost_1_expr_129 , 0 )
 GC780: Fraction[130] = LOGISTIC ( f_var[130] )
 min_sales_constr_130: Sold_0[130] = MIN ( Q0_var[130] , 1141.787276912816 )
 max_lost_constr_130: Lost_0[130] = MAX ( Lost_0_expr_130 , 0 )
 max_Q1_constr_130: Q1_var[130] = MAX ( Q_hat_adjusted_130 , 0 )
 min_sales2_constr_130: Sold_1[130] = MIN ( total_stock_second_period_130
   , 209.40249244818 )
 max_lost2_constr_130: Lost_1[130] = MAX ( Lost_1_expr_130 , 0 )
 GC786: Fraction[131] = LOGISTIC ( f_var[131] )
 min_sales_constr_131: Sold_0[131] = MIN ( Q0_var[131] , 2318.382377519079 )
 max_lost_constr_131: Lost_0[131] = MAX ( Lost_0_expr_131 , 0 )
 max_Q1_constr_131: Q1_var[131] = MAX ( Q_hat_adjusted_131 , 0 )
 min_sales2_constr_131: Sold_1[131] = MIN ( total_stock_second_period_131
   , 574.501528009008 )
 max_lost2_constr_131: Lost_1[131] = MAX ( Lost_1_expr_131 , 0 )
 GC792: Fraction[132] = LOGISTIC ( f_var[132] )
 min_sales_constr_132: Sold_0[132] = MIN ( Q0_var[132] , 966.2327486166023 )
 max_lost_constr_132: Lost_0[132] = MAX ( Lost_0_expr_132 , 0 )
 max_Q1_constr_132: Q1_var[132] = MAX ( Q_hat_adjusted_132 , 0 )
 min_sales2_constr_132: Sold_1[132] = MIN ( total_stock_second_period_132
   , 217.6407892664828 )
 max_lost2_constr_132: Lost_1[132] = MAX ( Lost_1_expr_132 , 0 )
 GC798: Fraction[133] = LOGISTIC ( f_var[133] )
 min_sales_constr_133: Sold_0[133] = MIN ( Q0_var[133] , 973.7608047145944 )
 max_lost_constr_133: Lost_0[133] = MAX ( Lost_0_expr_133 , 0 )
 max_Q1_constr_133: Q1_var[133] = MAX ( Q_hat_adjusted_133 , 0 )
 min_sales2_constr_133: Sold_1[133] = MIN ( total_stock_second_period_133
   , 254.5005201784956 )
 max_lost2_constr_133: Lost_1[133] = MAX ( Lost_1_expr_133 , 0 )
 GC804: Fraction[134] = LOGISTIC ( f_var[134] )
 min_sales_constr_134: Sold_0[134] = MIN ( Q0_var[134] , 1376.996051280214 )
 max_lost_constr_134: Lost_0[134] = MAX ( Lost_0_expr_134 , 0 )
 max_Q1_constr_134: Q1_var[134] = MAX ( Q_hat_adjusted_134 , 0 )
 min_sales2_constr_134: Sold_1[134] = MIN ( total_stock_second_period_134
   , 320.9160247059754 )
 max_lost2_constr_134: Lost_1[134] = MAX ( Lost_1_expr_134 , 0 )
 GC810: Fraction[135] = LOGISTIC ( f_var[135] )
 min_sales_constr_135: Sold_0[135] = MIN ( Q0_var[135] , 1302.516793859506 )
 max_lost_constr_135: Lost_0[135] = MAX ( Lost_0_expr_135 , 0 )
 max_Q1_constr_135: Q1_var[135] = MAX ( Q_hat_adjusted_135 , 0 )
 min_sales2_constr_135: Sold_1[135] = MIN ( total_stock_second_period_135
   , 360.3649742202745 )
 max_lost2_constr_135: Lost_1[135] = MAX ( Lost_1_expr_135 , 0 )
 GC816: Fraction[136] = LOGISTIC ( f_var[136] )
 min_sales_constr_136: Sold_0[136] = MIN ( Q0_var[136] , 1505.699408066973 )
 max_lost_constr_136: Lost_0[136] = MAX ( Lost_0_expr_136 , 0 )
 max_Q1_constr_136: Q1_var[136] = MAX ( Q_hat_adjusted_136 , 0 )
 min_sales2_constr_136: Sold_1[136] = MIN ( total_stock_second_period_136
   , 327.758661065081 )
 max_lost2_constr_136: Lost_1[136] = MAX ( Lost_1_expr_136 , 0 )
 GC822: Fraction[137] = LOGISTIC ( f_var[137] )
 min_sales_constr_137: Sold_0[137] = MIN ( Q0_var[137] , 538.5244300286142 )
 max_lost_constr_137: Lost_0[137] = MAX ( Lost_0_expr_137 , 0 )
 max_Q1_constr_137: Q1_var[137] = MAX ( Q_hat_adjusted_137 , 0 )
 min_sales2_constr_137: Sold_1[137] = MIN ( total_stock_second_period_137
   , 100.1034742410803 )
 max_lost2_constr_137: Lost_1[137] = MAX ( Lost_1_expr_137 , 0 )
 GC828: Fraction[138] = LOGISTIC ( f_var[138] )
 min_sales_constr_138: Sold_0[138] = MIN ( Q0_var[138] , 2357.305547418276 )
 max_lost_constr_138: Lost_0[138] = MAX ( Lost_0_expr_138 , 0 )
 max_Q1_constr_138: Q1_var[138] = MAX ( Q_hat_adjusted_138 , 0 )
 min_sales2_constr_138: Sold_1[138] = MIN ( total_stock_second_period_138
   , 531.0982298471089 )
 max_lost2_constr_138: Lost_1[138] = MAX ( Lost_1_expr_138 , 0 )
 GC834: Fraction[139] = LOGISTIC ( f_var[139] )
 min_sales_constr_139: Sold_0[139] = MIN ( Q0_var[139] , 645.2420193336925 )
 max_lost_constr_139: Lost_0[139] = MAX ( Lost_0_expr_139 , 0 )
 max_Q1_constr_139: Q1_var[139] = MAX ( Q_hat_adjusted_139 , 0 )
 min_sales2_constr_139: Sold_1[139] = MIN ( total_stock_second_period_139
   , 111.8693107045134 )
 max_lost2_constr_139: Lost_1[139] = MAX ( Lost_1_expr_139 , 0 )
 GC840: Fraction[140] = LOGISTIC ( f_var[140] )
 min_sales_constr_140: Sold_0[140] = MIN ( Q0_var[140] , 2100.821620038143 )
 max_lost_constr_140: Lost_0[140] = MAX ( Lost_0_expr_140 , 0 )
 max_Q1_constr_140: Q1_var[140] = MAX ( Q_hat_adjusted_140 , 0 )
 min_sales2_constr_140: Sold_1[140] = MIN ( total_stock_second_period_140
   , 540.7753049512376 )
 max_lost2_constr_140: Lost_1[140] = MAX ( Lost_1_expr_140 , 0 )
 GC846: Fraction[141] = LOGISTIC ( f_var[141] )
 min_sales_constr_141: Sold_0[141] = MIN ( Q0_var[141] , 2414.018727910669 )
 max_lost_constr_141: Lost_0[141] = MAX ( Lost_0_expr_141 , 0 )
 max_Q1_constr_141: Q1_var[141] = MAX ( Q_hat_adjusted_141 , 0 )
 min_sales2_constr_141: Sold_1[141] = MIN ( total_stock_second_period_141
   , 511.4088725225142 )
 max_lost2_constr_141: Lost_1[141] = MAX ( Lost_1_expr_141 , 0 )
 GC852: Fraction[142] = LOGISTIC ( f_var[142] )
 min_sales_constr_142: Sold_0[142] = MIN ( Q0_var[142] , 2090.234860729448 )
 max_lost_constr_142: Lost_0[142] = MAX ( Lost_0_expr_142 , 0 )
 max_Q1_constr_142: Q1_var[142] = MAX ( Q_hat_adjusted_142 , 0 )
 min_sales2_constr_142: Sold_1[142] = MIN ( total_stock_second_period_142
   , 568.3495539321655 )
 max_lost2_constr_142: Lost_1[142] = MAX ( Lost_1_expr_142 , 0 )
 GC858: Fraction[143] = LOGISTIC ( f_var[143] )
 min_sales_constr_143: Sold_0[143] = MIN ( Q0_var[143] , 1333.484681078238 )
 max_lost_constr_143: Lost_0[143] = MAX ( Lost_0_expr_143 , 0 )
 max_Q1_constr_143: Q1_var[143] = MAX ( Q_hat_adjusted_143 , 0 )
 min_sales2_constr_143: Sold_1[143] = MIN ( total_stock_second_period_143
   , 256.9817250913162 )
 max_lost2_constr_143: Lost_1[143] = MAX ( Lost_1_expr_143 , 0 )
 GC864: Fraction[144] = LOGISTIC ( f_var[144] )
 min_sales_constr_144: Sold_0[144] = MIN ( Q0_var[144] , 767.4414383373116 )
 max_lost_constr_144: Lost_0[144] = MAX ( Lost_0_expr_144 , 0 )
 max_Q1_constr_144: Q1_var[144] = MAX ( Q_hat_adjusted_144 , 0 )
 min_sales2_constr_144: Sold_1[144] = MIN ( total_stock_second_period_144
   , 189.8534998531855 )
 max_lost2_constr_144: Lost_1[144] = MAX ( Lost_1_expr_144 , 0 )
 GC870: Fraction[145] = LOGISTIC ( f_var[145] )
 min_sales_constr_145: Sold_0[145] = MIN ( Q0_var[145] , 635.0377301562764 )
 max_lost_constr_145: Lost_0[145] = MAX ( Lost_0_expr_145 , 0 )
 max_Q1_constr_145: Q1_var[145] = MAX ( Q_hat_adjusted_145 , 0 )
 min_sales2_constr_145: Sold_1[145] = MIN ( total_stock_second_period_145
   , 127.143263120872 )
 max_lost2_constr_145: Lost_1[145] = MAX ( Lost_1_expr_145 , 0 )
 GC876: Fraction[146] = LOGISTIC ( f_var[146] )
 min_sales_constr_146: Sold_0[146] = MIN ( Q0_var[146] , 1488.438331426776 )
 max_lost_constr_146: Lost_0[146] = MAX ( Lost_0_expr_146 , 0 )
 max_Q1_constr_146: Q1_var[146] = MAX ( Q_hat_adjusted_146 , 0 )
 min_sales2_constr_146: Sold_1[146] = MIN ( total_stock_second_period_146
   , 443.8713484625748 )
 max_lost2_constr_146: Lost_1[146] = MAX ( Lost_1_expr_146 , 0 )
 GC882: Fraction[147] = LOGISTIC ( f_var[147] )
 min_sales_constr_147: Sold_0[147] = MIN ( Q0_var[147] , 631.0970486458461 )
 max_lost_constr_147: Lost_0[147] = MAX ( Lost_0_expr_147 , 0 )
 max_Q1_constr_147: Q1_var[147] = MAX ( Q_hat_adjusted_147 , 0 )
 min_sales2_constr_147: Sold_1[147] = MIN ( total_stock_second_period_147
   , 178.2794257762002 )
 max_lost2_constr_147: Lost_1[147] = MAX ( Lost_1_expr_147 , 0 )
 GC888: Fraction[148] = LOGISTIC ( f_var[148] )
 min_sales_constr_148: Sold_0[148] = MIN ( Q0_var[148] , 859.5923337438492 )
 max_lost_constr_148: Lost_0[148] = MAX ( Lost_0_expr_148 , 0 )
 max_Q1_constr_148: Q1_var[148] = MAX ( Q_hat_adjusted_148 , 0 )
 min_sales2_constr_148: Sold_1[148] = MIN ( total_stock_second_period_148
   , 183.9157423590286 )
 max_lost2_constr_148: Lost_1[148] = MAX ( Lost_1_expr_148 , 0 )
 GC894: Fraction[149] = LOGISTIC ( f_var[149] )
 min_sales_constr_149: Sold_0[149] = MIN ( Q0_var[149] , 1241.228633023082 )
 max_lost_constr_149: Lost_0[149] = MAX ( Lost_0_expr_149 , 0 )
 max_Q1_constr_149: Q1_var[149] = MAX ( Q_hat_adjusted_149 , 0 )
 min_sales2_constr_149: Sold_1[149] = MIN ( total_stock_second_period_149
   , 215.1964609164928 )
 max_lost2_constr_149: Lost_1[149] = MAX ( Lost_1_expr_149 , 0 )
 GC900: Fraction[150] = LOGISTIC ( f_var[150] )
 min_sales_constr_150: Sold_0[150] = MIN ( Q0_var[150] , 799.5137118898315 )
 max_lost_constr_150: Lost_0[150] = MAX ( Lost_0_expr_150 , 0 )
 max_Q1_constr_150: Q1_var[150] = MAX ( Q_hat_adjusted_150 , 0 )
 min_sales2_constr_150: Sold_1[150] = MIN ( total_stock_second_period_150
   , 239.3095294532166 )
 max_lost2_constr_150: Lost_1[150] = MAX ( Lost_1_expr_150 , 0 )
 GC906: Fraction[151] = LOGISTIC ( f_var[151] )
 min_sales_constr_151: Sold_0[151] = MIN ( Q0_var[151] , 911.3885649941496 )
 max_lost_constr_151: Lost_0[151] = MAX ( Lost_0_expr_151 , 0 )
 max_Q1_constr_151: Q1_var[151] = MAX ( Q_hat_adjusted_151 , 0 )
 min_sales2_constr_151: Sold_1[151] = MIN ( total_stock_second_period_151
   , 206.7982921064874 )
 max_lost2_constr_151: Lost_1[151] = MAX ( Lost_1_expr_151 , 0 )
 GC912: Fraction[152] = LOGISTIC ( f_var[152] )
 min_sales_constr_152: Sold_0[152] = MIN ( Q0_var[152] , 1615.231555467062 )
 max_lost_constr_152: Lost_0[152] = MAX ( Lost_0_expr_152 , 0 )
 max_Q1_constr_152: Q1_var[152] = MAX ( Q_hat_adjusted_152 , 0 )
 min_sales2_constr_152: Sold_1[152] = MIN ( total_stock_second_period_152
   , 403.7086176181698 )
 max_lost2_constr_152: Lost_1[152] = MAX ( Lost_1_expr_152 , 0 )
 GC918: Fraction[153] = LOGISTIC ( f_var[153] )
 min_sales_constr_153: Sold_0[153] = MIN ( Q0_var[153] , 1829.154445955313 )
 max_lost_constr_153: Lost_0[153] = MAX ( Lost_0_expr_153 , 0 )
 max_Q1_constr_153: Q1_var[153] = MAX ( Q_hat_adjusted_153 , 0 )
 min_sales2_constr_153: Sold_1[153] = MIN ( total_stock_second_period_153
   , 355.5579716832476 )
 max_lost2_constr_153: Lost_1[153] = MAX ( Lost_1_expr_153 , 0 )
 GC924: Fraction[154] = LOGISTIC ( f_var[154] )
 min_sales_constr_154: Sold_0[154] = MIN ( Q0_var[154] , 1470.639677097006 )
 max_lost_constr_154: Lost_0[154] = MAX ( Lost_0_expr_154 , 0 )
 max_Q1_constr_154: Q1_var[154] = MAX ( Q_hat_adjusted_154 , 0 )
 min_sales2_constr_154: Sold_1[154] = MIN ( total_stock_second_period_154
   , 425.3432957477161 )
 max_lost2_constr_154: Lost_1[154] = MAX ( Lost_1_expr_154 , 0 )
 GC930: Fraction[155] = LOGISTIC ( f_var[155] )
 min_sales_constr_155: Sold_0[155] = MIN ( Q0_var[155] , 1584.510587266324 )
 max_lost_constr_155: Lost_0[155] = MAX ( Lost_0_expr_155 , 0 )
 max_Q1_constr_155: Q1_var[155] = MAX ( Q_hat_adjusted_155 , 0 )
 min_sales2_constr_155: Sold_1[155] = MIN ( total_stock_second_period_155
   , 421.0241077352973 )
 max_lost2_constr_155: Lost_1[155] = MAX ( Lost_1_expr_155 , 0 )
 GC936: Fraction[156] = LOGISTIC ( f_var[156] )
 min_sales_constr_156: Sold_0[156] = MIN ( Q0_var[156] , 2091.059538069346 )
 max_lost_constr_156: Lost_0[156] = MAX ( Lost_0_expr_156 , 0 )
 max_Q1_constr_156: Q1_var[156] = MAX ( Q_hat_adjusted_156 , 0 )
 min_sales2_constr_156: Sold_1[156] = MIN ( total_stock_second_period_156
   , 527.5778780921252 )
 max_lost2_constr_156: Lost_1[156] = MAX ( Lost_1_expr_156 , 0 )
 GC942: Fraction[157] = LOGISTIC ( f_var[157] )
 min_sales_constr_157: Sold_0[157] = MIN ( Q0_var[157] , 759.3635603841593 )
 max_lost_constr_157: Lost_0[157] = MAX ( Lost_0_expr_157 , 0 )
 max_Q1_constr_157: Q1_var[157] = MAX ( Q_hat_adjusted_157 , 0 )
 min_sales2_constr_157: Sold_1[157] = MIN ( total_stock_second_period_157
   , 118.3633260288806 )
 max_lost2_constr_157: Lost_1[157] = MAX ( Lost_1_expr_157 , 0 )
 GC948: Fraction[158] = LOGISTIC ( f_var[158] )
 min_sales_constr_158: Sold_0[158] = MIN ( Q0_var[158] , 1470.373620015705 )
 max_lost_constr_158: Lost_0[158] = MAX ( Lost_0_expr_158 , 0 )
 max_Q1_constr_158: Q1_var[158] = MAX ( Q_hat_adjusted_158 , 0 )
 min_sales2_constr_158: Sold_1[158] = MIN ( total_stock_second_period_158
   , 378.8558300336417 )
 max_lost2_constr_158: Lost_1[158] = MAX ( Lost_1_expr_158 , 0 )
 GC954: Fraction[159] = LOGISTIC ( f_var[159] )
 min_sales_constr_159: Sold_0[159] = MIN ( Q0_var[159] , 1323.733607878937 )
 max_lost_constr_159: Lost_0[159] = MAX ( Lost_0_expr_159 , 0 )
 max_Q1_constr_159: Q1_var[159] = MAX ( Q_hat_adjusted_159 , 0 )
 min_sales2_constr_159: Sold_1[159] = MIN ( total_stock_second_period_159
   , 292.9707949336987 )
 max_lost2_constr_159: Lost_1[159] = MAX ( Lost_1_expr_159 , 0 )
 GC960: Fraction[160] = LOGISTIC ( f_var[160] )
 min_sales_constr_160: Sold_0[160] = MIN ( Q0_var[160] , 2206.964853220385 )
 max_lost_constr_160: Lost_0[160] = MAX ( Lost_0_expr_160 , 0 )
 max_Q1_constr_160: Q1_var[160] = MAX ( Q_hat_adjusted_160 , 0 )
 min_sales2_constr_160: Sold_1[160] = MIN ( total_stock_second_period_160
   , 494.6323325756994 )
 max_lost2_constr_160: Lost_1[160] = MAX ( Lost_1_expr_160 , 0 )
 GC966: Fraction[161] = LOGISTIC ( f_var[161] )
 min_sales_constr_161: Sold_0[161] = MIN ( Q0_var[161] , 759.550583475306 )
 max_lost_constr_161: Lost_0[161] = MAX ( Lost_0_expr_161 , 0 )
 max_Q1_constr_161: Q1_var[161] = MAX ( Q_hat_adjusted_161 , 0 )
 min_sales2_constr_161: Sold_1[161] = MIN ( total_stock_second_period_161
   , 214.2773740980794 )
 max_lost2_constr_161: Lost_1[161] = MAX ( Lost_1_expr_161 , 0 )
 GC972: Fraction[162] = LOGISTIC ( f_var[162] )
 min_sales_constr_162: Sold_0[162] = MIN ( Q0_var[162] , 1457.705769615989 )
 max_lost_constr_162: Lost_0[162] = MAX ( Lost_0_expr_162 , 0 )
 max_Q1_constr_162: Q1_var[162] = MAX ( Q_hat_adjusted_162 , 0 )
 min_sales2_constr_162: Sold_1[162] = MIN ( total_stock_second_period_162
   , 404.47585948292 )
 max_lost2_constr_162: Lost_1[162] = MAX ( Lost_1_expr_162 , 0 )
 GC978: Fraction[163] = LOGISTIC ( f_var[163] )
 min_sales_constr_163: Sold_0[163] = MIN ( Q0_var[163] , 763.0501638787257 )
 max_lost_constr_163: Lost_0[163] = MAX ( Lost_0_expr_163 , 0 )
 max_Q1_constr_163: Q1_var[163] = MAX ( Q_hat_adjusted_163 , 0 )
 min_sales2_constr_163: Sold_1[163] = MIN ( total_stock_second_period_163
   , 234.7967120117548 )
 max_lost2_constr_163: Lost_1[163] = MAX ( Lost_1_expr_163 , 0 )
 GC984: Fraction[164] = LOGISTIC ( f_var[164] )
 min_sales_constr_164: Sold_0[164] = MIN ( Q0_var[164] , 800.4750615408716 )
 max_lost_constr_164: Lost_0[164] = MAX ( Lost_0_expr_164 , 0 )
 max_Q1_constr_164: Q1_var[164] = MAX ( Q_hat_adjusted_164 , 0 )
 min_sales2_constr_164: Sold_1[164] = MIN ( total_stock_second_period_164
   , 217.3355924152405 )
 max_lost2_constr_164: Lost_1[164] = MAX ( Lost_1_expr_164 , 0 )
 GC990: Fraction[165] = LOGISTIC ( f_var[165] )
 min_sales_constr_165: Sold_0[165] = MIN ( Q0_var[165] , 676.6640105223669 )
 max_lost_constr_165: Lost_0[165] = MAX ( Lost_0_expr_165 , 0 )
 max_Q1_constr_165: Q1_var[165] = MAX ( Q_hat_adjusted_165 , 0 )
 min_sales2_constr_165: Sold_1[165] = MIN ( total_stock_second_period_165
   , 263.9184925317116 )
 max_lost2_constr_165: Lost_1[165] = MAX ( Lost_1_expr_165 , 0 )
 GC996: Fraction[166] = LOGISTIC ( f_var[166] )
 min_sales_constr_166: Sold_0[166] = MIN ( Q0_var[166] , 542.5703198251656 )
 max_lost_constr_166: Lost_0[166] = MAX ( Lost_0_expr_166 , 0 )
 max_Q1_constr_166: Q1_var[166] = MAX ( Q_hat_adjusted_166 , 0 )
 min_sales2_constr_166: Sold_1[166] = MIN ( total_stock_second_period_166
   , 131.7878706435609 )
 max_lost2_constr_166: Lost_1[166] = MAX ( Lost_1_expr_166 , 0 )
 GC1002: Fraction[167] = LOGISTIC ( f_var[167] )
 min_sales_constr_167: Sold_0[167] = MIN ( Q0_var[167] , 1739.078625978894 )
 max_lost_constr_167: Lost_0[167] = MAX ( Lost_0_expr_167 , 0 )
 max_Q1_constr_167: Q1_var[167] = MAX ( Q_hat_adjusted_167 , 0 )
 min_sales2_constr_167: Sold_1[167] = MIN ( total_stock_second_period_167
   , 473.1931139358691 )
 max_lost2_constr_167: Lost_1[167] = MAX ( Lost_1_expr_167 , 0 )
 GC1008: Fraction[168] = LOGISTIC ( f_var[168] )
 min_sales_constr_168: Sold_0[168] = MIN ( Q0_var[168] , 1788.198082364279 )
 max_lost_constr_168: Lost_0[168] = MAX ( Lost_0_expr_168 , 0 )
 max_Q1_constr_168: Q1_var[168] = MAX ( Q_hat_adjusted_168 , 0 )
 min_sales2_constr_168: Sold_1[168] = MIN ( total_stock_second_period_168
   , 540.6550302353056 )
 max_lost2_constr_168: Lost_1[168] = MAX ( Lost_1_expr_168 , 0 )
 GC1014: Fraction[169] = LOGISTIC ( f_var[169] )
 min_sales_constr_169: Sold_0[169] = MIN ( Q0_var[169] , 744.5910617144899 )
 max_lost_constr_169: Lost_0[169] = MAX ( Lost_0_expr_169 , 0 )
 max_Q1_constr_169: Q1_var[169] = MAX ( Q_hat_adjusted_169 , 0 )
 min_sales2_constr_169: Sold_1[169] = MIN ( total_stock_second_period_169
   , 108.4925547459586 )
 max_lost2_constr_169: Lost_1[169] = MAX ( Lost_1_expr_169 , 0 )
 GC1020: Fraction[170] = LOGISTIC ( f_var[170] )
 min_sales_constr_170: Sold_0[170] = MIN ( Q0_var[170] , 1767.838506631978 )
 max_lost_constr_170: Lost_0[170] = MAX ( Lost_0_expr_170 , 0 )
 max_Q1_constr_170: Q1_var[170] = MAX ( Q_hat_adjusted_170 , 0 )
 min_sales2_constr_170: Sold_1[170] = MIN ( total_stock_second_period_170
   , 426.221207570495 )
 max_lost2_constr_170: Lost_1[170] = MAX ( Lost_1_expr_170 , 0 )
 GC1026: Fraction[171] = LOGISTIC ( f_var[171] )
 min_sales_constr_171: Sold_0[171] = MIN ( Q0_var[171] , 1545.099074270043 )
 max_lost_constr_171: Lost_0[171] = MAX ( Lost_0_expr_171 , 0 )
 max_Q1_constr_171: Q1_var[171] = MAX ( Q_hat_adjusted_171 , 0 )
 min_sales2_constr_171: Sold_1[171] = MIN ( total_stock_second_period_171
   , 381.4503314616078 )
 max_lost2_constr_171: Lost_1[171] = MAX ( Lost_1_expr_171 , 0 )
 GC1032: Fraction[172] = LOGISTIC ( f_var[172] )
 min_sales_constr_172: Sold_0[172] = MIN ( Q0_var[172] , 2237.910684109235 )
 max_lost_constr_172: Lost_0[172] = MAX ( Lost_0_expr_172 , 0 )
 max_Q1_constr_172: Q1_var[172] = MAX ( Q_hat_adjusted_172 , 0 )
 min_sales2_constr_172: Sold_1[172] = MIN ( total_stock_second_period_172
   , 552.6969501088421 )
 max_lost2_constr_172: Lost_1[172] = MAX ( Lost_1_expr_172 , 0 )
 GC1038: Fraction[173] = LOGISTIC ( f_var[173] )
 min_sales_constr_173: Sold_0[173] = MIN ( Q0_var[173] , 2171.079233521988 )
 max_lost_constr_173: Lost_0[173] = MAX ( Lost_0_expr_173 , 0 )
 max_Q1_constr_173: Q1_var[173] = MAX ( Q_hat_adjusted_173 , 0 )
 min_sales2_constr_173: Sold_1[173] = MIN ( total_stock_second_period_173
   , 442.6801133675785 )
 max_lost2_constr_173: Lost_1[173] = MAX ( Lost_1_expr_173 , 0 )
 GC1044: Fraction[174] = LOGISTIC ( f_var[174] )
 min_sales_constr_174: Sold_0[174] = MIN ( Q0_var[174] , 1086.354929764516 )
 max_lost_constr_174: Lost_0[174] = MAX ( Lost_0_expr_174 , 0 )
 max_Q1_constr_174: Q1_var[174] = MAX ( Q_hat_adjusted_174 , 0 )
 min_sales2_constr_174: Sold_1[174] = MIN ( total_stock_second_period_174
   , 309.292487546168 )
 max_lost2_constr_174: Lost_1[174] = MAX ( Lost_1_expr_174 , 0 )
 GC1050: Fraction[175] = LOGISTIC ( f_var[175] )
 min_sales_constr_175: Sold_0[175] = MIN ( Q0_var[175] , 1679.069769820995 )
 max_lost_constr_175: Lost_0[175] = MAX ( Lost_0_expr_175 , 0 )
 max_Q1_constr_175: Q1_var[175] = MAX ( Q_hat_adjusted_175 , 0 )
 min_sales2_constr_175: Sold_1[175] = MIN ( total_stock_second_period_175
   , 389.0980220136706 )
 max_lost2_constr_175: Lost_1[175] = MAX ( Lost_1_expr_175 , 0 )
 GC1056: Fraction[176] = LOGISTIC ( f_var[176] )
 min_sales_constr_176: Sold_0[176] = MIN ( Q0_var[176] , 615.1309393411314 )
 max_lost_constr_176: Lost_0[176] = MAX ( Lost_0_expr_176 , 0 )
 max_Q1_constr_176: Q1_var[176] = MAX ( Q_hat_adjusted_176 , 0 )
 min_sales2_constr_176: Sold_1[176] = MIN ( total_stock_second_period_176
   , 163.6072421599445 )
 max_lost2_constr_176: Lost_1[176] = MAX ( Lost_1_expr_176 , 0 )
 GC1062: Fraction[177] = LOGISTIC ( f_var[177] )
 min_sales_constr_177: Sold_0[177] = MIN ( Q0_var[177] , 642.6433376589987 )
 max_lost_constr_177: Lost_0[177] = MAX ( Lost_0_expr_177 , 0 )
 max_Q1_constr_177: Q1_var[177] = MAX ( Q_hat_adjusted_177 , 0 )
 min_sales2_constr_177: Sold_1[177] = MIN ( total_stock_second_period_177
   , 181.0750601305231 )
 max_lost2_constr_177: Lost_1[177] = MAX ( Lost_1_expr_177 , 0 )
 GC1068: Fraction[178] = LOGISTIC ( f_var[178] )
 min_sales_constr_178: Sold_0[178] = MIN ( Q0_var[178] , 1228.080107464607 )
 max_lost_constr_178: Lost_0[178] = MAX ( Lost_0_expr_178 , 0 )
 max_Q1_constr_178: Q1_var[178] = MAX ( Q_hat_adjusted_178 , 0 )
 min_sales2_constr_178: Sold_1[178] = MIN ( total_stock_second_period_178
   , 211.9184731654302 )
 max_lost2_constr_178: Lost_1[178] = MAX ( Lost_1_expr_178 , 0 )
 GC1074: Fraction[179] = LOGISTIC ( f_var[179] )
 min_sales_constr_179: Sold_0[179] = MIN ( Q0_var[179] , 1274.396032361624 )
 max_lost_constr_179: Lost_0[179] = MAX ( Lost_0_expr_179 , 0 )
 max_Q1_constr_179: Q1_var[179] = MAX ( Q_hat_adjusted_179 , 0 )
 min_sales2_constr_179: Sold_1[179] = MIN ( total_stock_second_period_179
   , 341.5450470805698 )
 max_lost2_constr_179: Lost_1[179] = MAX ( Lost_1_expr_179 , 0 )
 GC1080: Fraction[180] = LOGISTIC ( f_var[180] )
 min_sales_constr_180: Sold_0[180] = MIN ( Q0_var[180] , 1048.111424598424 )
 max_lost_constr_180: Lost_0[180] = MAX ( Lost_0_expr_180 , 0 )
 max_Q1_constr_180: Q1_var[180] = MAX ( Q_hat_adjusted_180 , 0 )
 min_sales2_constr_180: Sold_1[180] = MIN ( total_stock_second_period_180
   , 233.4120553476735 )
 max_lost2_constr_180: Lost_1[180] = MAX ( Lost_1_expr_180 , 0 )
 GC1086: Fraction[181] = LOGISTIC ( f_var[181] )
 min_sales_constr_181: Sold_0[181] = MIN ( Q0_var[181] , 606.5252479629278 )
 max_lost_constr_181: Lost_0[181] = MAX ( Lost_0_expr_181 , 0 )
 max_Q1_constr_181: Q1_var[181] = MAX ( Q_hat_adjusted_181 , 0 )
 min_sales2_constr_181: Sold_1[181] = MIN ( total_stock_second_period_181
   , 150.2721942986757 )
 max_lost2_constr_181: Lost_1[181] = MAX ( Lost_1_expr_181 , 0 )
 GC1092: Fraction[182] = LOGISTIC ( f_var[182] )
 min_sales_constr_182: Sold_0[182] = MIN ( Q0_var[182] , 1548.402021008132 )
 max_lost_constr_182: Lost_0[182] = MAX ( Lost_0_expr_182 , 0 )
 max_Q1_constr_182: Q1_var[182] = MAX ( Q_hat_adjusted_182 , 0 )
 min_sales2_constr_182: Sold_1[182] = MIN ( total_stock_second_period_182
   , 359.254223060561 )
 max_lost2_constr_182: Lost_1[182] = MAX ( Lost_1_expr_182 , 0 )
 GC1098: Fraction[183] = LOGISTIC ( f_var[183] )
 min_sales_constr_183: Sold_0[183] = MIN ( Q0_var[183] , 1966.01250220268 )
 max_lost_constr_183: Lost_0[183] = MAX ( Lost_0_expr_183 , 0 )
 max_Q1_constr_183: Q1_var[183] = MAX ( Q_hat_adjusted_183 , 0 )
 min_sales2_constr_183: Sold_1[183] = MIN ( total_stock_second_period_183
   , 448.9148855134819 )
 max_lost2_constr_183: Lost_1[183] = MAX ( Lost_1_expr_183 , 0 )
 GC1104: Fraction[184] = LOGISTIC ( f_var[184] )
 min_sales_constr_184: Sold_0[184] = MIN ( Q0_var[184] , 582.4441723661721 )
 max_lost_constr_184: Lost_0[184] = MAX ( Lost_0_expr_184 , 0 )
 max_Q1_constr_184: Q1_var[184] = MAX ( Q_hat_adjusted_184 , 0 )
 min_sales2_constr_184: Sold_1[184] = MIN ( total_stock_second_period_184
   , 148.219541181427 )
 max_lost2_constr_184: Lost_1[184] = MAX ( Lost_1_expr_184 , 0 )
 GC1110: Fraction[185] = LOGISTIC ( f_var[185] )
 min_sales_constr_185: Sold_0[185] = MIN ( Q0_var[185] , 2041.448617544953 )
 max_lost_constr_185: Lost_0[185] = MAX ( Lost_0_expr_185 , 0 )
 max_Q1_constr_185: Q1_var[185] = MAX ( Q_hat_adjusted_185 , 0 )
 min_sales2_constr_185: Sold_1[185] = MIN ( total_stock_second_period_185
   , 464.4764097326664 )
 max_lost2_constr_185: Lost_1[185] = MAX ( Lost_1_expr_185 , 0 )
 GC1116: Fraction[186] = LOGISTIC ( f_var[186] )
 min_sales_constr_186: Sold_0[186] = MIN ( Q0_var[186] , 359.1198295807582 )
 max_lost_constr_186: Lost_0[186] = MAX ( Lost_0_expr_186 , 0 )
 max_Q1_constr_186: Q1_var[186] = MAX ( Q_hat_adjusted_186 , 0 )
 min_sales2_constr_186: Sold_1[186] = MIN ( total_stock_second_period_186
   , 111.8340838540373 )
 max_lost2_constr_186: Lost_1[186] = MAX ( Lost_1_expr_186 , 0 )
 GC1122: Fraction[187] = LOGISTIC ( f_var[187] )
 min_sales_constr_187: Sold_0[187] = MIN ( Q0_var[187] , 1861.159359438815 )
 max_lost_constr_187: Lost_0[187] = MAX ( Lost_0_expr_187 , 0 )
 max_Q1_constr_187: Q1_var[187] = MAX ( Q_hat_adjusted_187 , 0 )
 min_sales2_constr_187: Sold_1[187] = MIN ( total_stock_second_period_187
   , 449.4703117095602 )
 max_lost2_constr_187: Lost_1[187] = MAX ( Lost_1_expr_187 , 0 )
 GC1128: Fraction[188] = LOGISTIC ( f_var[188] )
 min_sales_constr_188: Sold_0[188] = MIN ( Q0_var[188] , 2188.505075995322 )
 max_lost_constr_188: Lost_0[188] = MAX ( Lost_0_expr_188 , 0 )
 max_Q1_constr_188: Q1_var[188] = MAX ( Q_hat_adjusted_188 , 0 )
 min_sales2_constr_188: Sold_1[188] = MIN ( total_stock_second_period_188
   , 576.5426274432116 )
 max_lost2_constr_188: Lost_1[188] = MAX ( Lost_1_expr_188 , 0 )
 GC1134: Fraction[189] = LOGISTIC ( f_var[189] )
 min_sales_constr_189: Sold_0[189] = MIN ( Q0_var[189] , 1499.796336489446 )
 max_lost_constr_189: Lost_0[189] = MAX ( Lost_0_expr_189 , 0 )
 max_Q1_constr_189: Q1_var[189] = MAX ( Q_hat_adjusted_189 , 0 )
 min_sales2_constr_189: Sold_1[189] = MIN ( total_stock_second_period_189
   , 328.3834303194924 )
 max_lost2_constr_189: Lost_1[189] = MAX ( Lost_1_expr_189 , 0 )
 GC1140: Fraction[190] = LOGISTIC ( f_var[190] )
 min_sales_constr_190: Sold_0[190] = MIN ( Q0_var[190] , 593.9709921442616 )
 max_lost_constr_190: Lost_0[190] = MAX ( Lost_0_expr_190 , 0 )
 max_Q1_constr_190: Q1_var[190] = MAX ( Q_hat_adjusted_190 , 0 )
 min_sales2_constr_190: Sold_1[190] = MIN ( total_stock_second_period_190
   , 132.9453786759889 )
 max_lost2_constr_190: Lost_1[190] = MAX ( Lost_1_expr_190 , 0 )
 GC1146: Fraction[191] = LOGISTIC ( f_var[191] )
 min_sales_constr_191: Sold_0[191] = MIN ( Q0_var[191] , 1000.260275127421 )
 max_lost_constr_191: Lost_0[191] = MAX ( Lost_0_expr_191 , 0 )
 max_Q1_constr_191: Q1_var[191] = MAX ( Q_hat_adjusted_191 , 0 )
 min_sales2_constr_191: Sold_1[191] = MIN ( total_stock_second_period_191
   , 241.3410293382373 )
 max_lost2_constr_191: Lost_1[191] = MAX ( Lost_1_expr_191 , 0 )
 GC1152: Fraction[192] = LOGISTIC ( f_var[192] )
 min_sales_constr_192: Sold_0[192] = MIN ( Q0_var[192] , 442.074196588126 )
 max_lost_constr_192: Lost_0[192] = MAX ( Lost_0_expr_192 , 0 )
 max_Q1_constr_192: Q1_var[192] = MAX ( Q_hat_adjusted_192 , 0 )
 min_sales2_constr_192: Sold_1[192] = MIN ( total_stock_second_period_192
   , 173.756906649892 )
 max_lost2_constr_192: Lost_1[192] = MAX ( Lost_1_expr_192 , 0 )
 GC1158: Fraction[193] = LOGISTIC ( f_var[193] )
 min_sales_constr_193: Sold_0[193] = MIN ( Q0_var[193] , 1548.201342472243 )
 max_lost_constr_193: Lost_0[193] = MAX ( Lost_0_expr_193 , 0 )
 max_Q1_constr_193: Q1_var[193] = MAX ( Q_hat_adjusted_193 , 0 )
 min_sales2_constr_193: Sold_1[193] = MIN ( total_stock_second_period_193
   , 350.4682867614392 )
 max_lost2_constr_193: Lost_1[193] = MAX ( Lost_1_expr_193 , 0 )
 GC1164: Fraction[194] = LOGISTIC ( f_var[194] )
 min_sales_constr_194: Sold_0[194] = MIN ( Q0_var[194] , 557.7831892918081 )
 max_lost_constr_194: Lost_0[194] = MAX ( Lost_0_expr_194 , 0 )
 max_Q1_constr_194: Q1_var[194] = MAX ( Q_hat_adjusted_194 , 0 )
 min_sales2_constr_194: Sold_1[194] = MIN ( total_stock_second_period_194
   , 190.4017236336499 )
 max_lost2_constr_194: Lost_1[194] = MAX ( Lost_1_expr_194 , 0 )
 GC1170: Fraction[195] = LOGISTIC ( f_var[195] )
 min_sales_constr_195: Sold_0[195] = MIN ( Q0_var[195] , 2453.201322462989 )
 max_lost_constr_195: Lost_0[195] = MAX ( Lost_0_expr_195 , 0 )
 max_Q1_constr_195: Q1_var[195] = MAX ( Q_hat_adjusted_195 , 0 )
 min_sales2_constr_195: Sold_1[195] = MIN ( total_stock_second_period_195
   , 598.1908490946607 )
 max_lost2_constr_195: Lost_1[195] = MAX ( Lost_1_expr_195 , 0 )
 GC1176: Fraction[196] = LOGISTIC ( f_var[196] )
 min_sales_constr_196: Sold_0[196] = MIN ( Q0_var[196] , 1324.405805391361 )
 max_lost_constr_196: Lost_0[196] = MAX ( Lost_0_expr_196 , 0 )
 max_Q1_constr_196: Q1_var[196] = MAX ( Q_hat_adjusted_196 , 0 )
 min_sales2_constr_196: Sold_1[196] = MIN ( total_stock_second_period_196
   , 309.7829020314259 )
 max_lost2_constr_196: Lost_1[196] = MAX ( Lost_1_expr_196 , 0 )
 GC1182: Fraction[197] = LOGISTIC ( f_var[197] )
 min_sales_constr_197: Sold_0[197] = MIN ( Q0_var[197] , 1166.596765455613 )
 max_lost_constr_197: Lost_0[197] = MAX ( Lost_0_expr_197 , 0 )
 max_Q1_constr_197: Q1_var[197] = MAX ( Q_hat_adjusted_197 , 0 )
 min_sales2_constr_197: Sold_1[197] = MIN ( total_stock_second_period_197
   , 293.6705934530557 )
 max_lost2_constr_197: Lost_1[197] = MAX ( Lost_1_expr_197 , 0 )
 GC1188: Fraction[198] = LOGISTIC ( f_var[198] )
 min_sales_constr_198: Sold_0[198] = MIN ( Q0_var[198] , 543.7053752641302 )
 max_lost_constr_198: Lost_0[198] = MAX ( Lost_0_expr_198 , 0 )
 max_Q1_constr_198: Q1_var[198] = MAX ( Q_hat_adjusted_198 , 0 )
 min_sales2_constr_198: Sold_1[198] = MIN ( total_stock_second_period_198
   , 132.0820433096775 )
 max_lost2_constr_198: Lost_1[198] = MAX ( Lost_1_expr_198 , 0 )
 GC1194: Fraction[199] = LOGISTIC ( f_var[199] )
 min_sales_constr_199: Sold_0[199] = MIN ( Q0_var[199] , 1837.44902778305 )
 max_lost_constr_199: Lost_0[199] = MAX ( Lost_0_expr_199 , 0 )
 max_Q1_constr_199: Q1_var[199] = MAX ( Q_hat_adjusted_199 , 0 )
 min_sales2_constr_199: Sold_1[199] = MIN ( total_stock_second_period_199
   , 399.2951090883211 )
 max_lost2_constr_199: Lost_1[199] = MAX ( Lost_1_expr_199 , 0 )
 GC1200: Fraction[200] = LOGISTIC ( f_var[200] )
 min_sales_constr_200: Sold_0[200] = MIN ( Q0_var[200] , 965.1500352182386 )
 max_lost_constr_200: Lost_0[200] = MAX ( Lost_0_expr_200 , 0 )
 max_Q1_constr_200: Q1_var[200] = MAX ( Q_hat_adjusted_200 , 0 )
 min_sales2_constr_200: Sold_1[200] = MIN ( total_stock_second_period_200
   , 162.1516613225906 )
 max_lost2_constr_200: Lost_1[200] = MAX ( Lost_1_expr_200 , 0 )
 GC1206: Fraction[201] = LOGISTIC ( f_var[201] )
 min_sales_constr_201: Sold_0[201] = MIN ( Q0_var[201] , 624.7340081762145 )
 max_lost_constr_201: Lost_0[201] = MAX ( Lost_0_expr_201 , 0 )
 max_Q1_constr_201: Q1_var[201] = MAX ( Q_hat_adjusted_201 , 0 )
 min_sales2_constr_201: Sold_1[201] = MIN ( total_stock_second_period_201
   , 230.8707326300203 )
 max_lost2_constr_201: Lost_1[201] = MAX ( Lost_1_expr_201 , 0 )
 GC1212: Fraction[202] = LOGISTIC ( f_var[202] )
 min_sales_constr_202: Sold_0[202] = MIN ( Q0_var[202] , 981.6673444654128 )
 max_lost_constr_202: Lost_0[202] = MAX ( Lost_0_expr_202 , 0 )
 max_Q1_constr_202: Q1_var[202] = MAX ( Q_hat_adjusted_202 , 0 )
 min_sales2_constr_202: Sold_1[202] = MIN ( total_stock_second_period_202
   , 249.5696242445653 )
 max_lost2_constr_202: Lost_1[202] = MAX ( Lost_1_expr_202 , 0 )
 GC1218: Fraction[203] = LOGISTIC ( f_var[203] )
 min_sales_constr_203: Sold_0[203] = MIN ( Q0_var[203] , 687.0287927576528 )
 max_lost_constr_203: Lost_0[203] = MAX ( Lost_0_expr_203 , 0 )
 max_Q1_constr_203: Q1_var[203] = MAX ( Q_hat_adjusted_203 , 0 )
 min_sales2_constr_203: Sold_1[203] = MIN ( total_stock_second_period_203
   , 149.0579959040795 )
 max_lost2_constr_203: Lost_1[203] = MAX ( Lost_1_expr_203 , 0 )
 GC1224: Fraction[204] = LOGISTIC ( f_var[204] )
 min_sales_constr_204: Sold_0[204] = MIN ( Q0_var[204] , 1773.78762113651 )
 max_lost_constr_204: Lost_0[204] = MAX ( Lost_0_expr_204 , 0 )
 max_Q1_constr_204: Q1_var[204] = MAX ( Q_hat_adjusted_204 , 0 )
 min_sales2_constr_204: Sold_1[204] = MIN ( total_stock_second_period_204
   , 515.0069193637528 )
 max_lost2_constr_204: Lost_1[204] = MAX ( Lost_1_expr_204 , 0 )
 GC1230: Fraction[205] = LOGISTIC ( f_var[205] )
 min_sales_constr_205: Sold_0[205] = MIN ( Q0_var[205] , 857.2230358368699 )
 max_lost_constr_205: Lost_0[205] = MAX ( Lost_0_expr_205 , 0 )
 max_Q1_constr_205: Q1_var[205] = MAX ( Q_hat_adjusted_205 , 0 )
 min_sales2_constr_205: Sold_1[205] = MIN ( total_stock_second_period_205
   , 155.2645052753531 )
 max_lost2_constr_205: Lost_1[205] = MAX ( Lost_1_expr_205 , 0 )
 GC1236: Fraction[206] = LOGISTIC ( f_var[206] )
 min_sales_constr_206: Sold_0[206] = MIN ( Q0_var[206] , 1209.444825216027 )
 max_lost_constr_206: Lost_0[206] = MAX ( Lost_0_expr_206 , 0 )
 max_Q1_constr_206: Q1_var[206] = MAX ( Q_hat_adjusted_206 , 0 )
 min_sales2_constr_206: Sold_1[206] = MIN ( total_stock_second_period_206
   , 341.5025155943215 )
 max_lost2_constr_206: Lost_1[206] = MAX ( Lost_1_expr_206 , 0 )
 GC1242: Fraction[207] = LOGISTIC ( f_var[207] )
 min_sales_constr_207: Sold_0[207] = MIN ( Q0_var[207] , 921.3868048682986 )
 max_lost_constr_207: Lost_0[207] = MAX ( Lost_0_expr_207 , 0 )
 max_Q1_constr_207: Q1_var[207] = MAX ( Q_hat_adjusted_207 , 0 )
 min_sales2_constr_207: Sold_1[207] = MIN ( total_stock_second_period_207
   , 244.5312898370325 )
 max_lost2_constr_207: Lost_1[207] = MAX ( Lost_1_expr_207 , 0 )
 GC1248: Fraction[208] = LOGISTIC ( f_var[208] )
 min_sales_constr_208: Sold_0[208] = MIN ( Q0_var[208] , 1051.578369570644 )
 max_lost_constr_208: Lost_0[208] = MAX ( Lost_0_expr_208 , 0 )
 max_Q1_constr_208: Q1_var[208] = MAX ( Q_hat_adjusted_208 , 0 )
 min_sales2_constr_208: Sold_1[208] = MIN ( total_stock_second_period_208
   , 218.2722828765191 )
 max_lost2_constr_208: Lost_1[208] = MAX ( Lost_1_expr_208 , 0 )
 GC1254: Fraction[209] = LOGISTIC ( f_var[209] )
 min_sales_constr_209: Sold_0[209] = MIN ( Q0_var[209] , 771.8647396875381 )
 max_lost_constr_209: Lost_0[209] = MAX ( Lost_0_expr_209 , 0 )
 max_Q1_constr_209: Q1_var[209] = MAX ( Q_hat_adjusted_209 , 0 )
 min_sales2_constr_209: Sold_1[209] = MIN ( total_stock_second_period_209
   , 211.6479264531634 )
 max_lost2_constr_209: Lost_1[209] = MAX ( Lost_1_expr_209 , 0 )
 GC1260: Fraction[210] = LOGISTIC ( f_var[210] )
 min_sales_constr_210: Sold_0[210] = MIN ( Q0_var[210] , 2119.948004618859 )
 max_lost_constr_210: Lost_0[210] = MAX ( Lost_0_expr_210 , 0 )
 max_Q1_constr_210: Q1_var[210] = MAX ( Q_hat_adjusted_210 , 0 )
 min_sales2_constr_210: Sold_1[210] = MIN ( total_stock_second_period_210
   , 434.9321879881747 )
 max_lost2_constr_210: Lost_1[210] = MAX ( Lost_1_expr_210 , 0 )
 GC1266: Fraction[211] = LOGISTIC ( f_var[211] )
 min_sales_constr_211: Sold_0[211] = MIN ( Q0_var[211] , 1723.497761392136 )
 max_lost_constr_211: Lost_0[211] = MAX ( Lost_0_expr_211 , 0 )
 max_Q1_constr_211: Q1_var[211] = MAX ( Q_hat_adjusted_211 , 0 )
 min_sales2_constr_211: Sold_1[211] = MIN ( total_stock_second_period_211
   , 496.8987525231301 )
 max_lost2_constr_211: Lost_1[211] = MAX ( Lost_1_expr_211 , 0 )
 GC1272: Fraction[212] = LOGISTIC ( f_var[212] )
 min_sales_constr_212: Sold_0[212] = MIN ( Q0_var[212] , 1030.501836563487 )
 max_lost_constr_212: Lost_0[212] = MAX ( Lost_0_expr_212 , 0 )
 max_Q1_constr_212: Q1_var[212] = MAX ( Q_hat_adjusted_212 , 0 )
 min_sales2_constr_212: Sold_1[212] = MIN ( total_stock_second_period_212
   , 304.4566791739243 )
 max_lost2_constr_212: Lost_1[212] = MAX ( Lost_1_expr_212 , 0 )
 GC1278: Fraction[213] = LOGISTIC ( f_var[213] )
 min_sales_constr_213: Sold_0[213] = MIN ( Q0_var[213] , 1203.44144579062 )
 max_lost_constr_213: Lost_0[213] = MAX ( Lost_0_expr_213 , 0 )
 max_Q1_constr_213: Q1_var[213] = MAX ( Q_hat_adjusted_213 , 0 )
 min_sales2_constr_213: Sold_1[213] = MIN ( total_stock_second_period_213
   , 259.2641516909673 )
 max_lost2_constr_213: Lost_1[213] = MAX ( Lost_1_expr_213 , 0 )
 GC1284: Fraction[214] = LOGISTIC ( f_var[214] )
 min_sales_constr_214: Sold_0[214] = MIN ( Q0_var[214] , 1818.825917399222 )
 max_lost_constr_214: Lost_0[214] = MAX ( Lost_0_expr_214 , 0 )
 max_Q1_constr_214: Q1_var[214] = MAX ( Q_hat_adjusted_214 , 0 )
 min_sales2_constr_214: Sold_1[214] = MIN ( total_stock_second_period_214
   , 465.5178056419056 )
 max_lost2_constr_214: Lost_1[214] = MAX ( Lost_1_expr_214 , 0 )
 GC1290: Fraction[215] = LOGISTIC ( f_var[215] )
 min_sales_constr_215: Sold_0[215] = MIN ( Q0_var[215] , 1086.400116806772 )
 max_lost_constr_215: Lost_0[215] = MAX ( Lost_0_expr_215 , 0 )
 max_Q1_constr_215: Q1_var[215] = MAX ( Q_hat_adjusted_215 , 0 )
 min_sales2_constr_215: Sold_1[215] = MIN ( total_stock_second_period_215
   , 234.5821156416472 )
 max_lost2_constr_215: Lost_1[215] = MAX ( Lost_1_expr_215 , 0 )
 GC1296: Fraction[216] = LOGISTIC ( f_var[216] )
 min_sales_constr_216: Sold_0[216] = MIN ( Q0_var[216] , 2024.545705082481 )
 max_lost_constr_216: Lost_0[216] = MAX ( Lost_0_expr_216 , 0 )
 max_Q1_constr_216: Q1_var[216] = MAX ( Q_hat_adjusted_216 , 0 )
 min_sales2_constr_216: Sold_1[216] = MIN ( total_stock_second_period_216
   , 535.2303694706526 )
 max_lost2_constr_216: Lost_1[216] = MAX ( Lost_1_expr_216 , 0 )
 GC1302: Fraction[217] = LOGISTIC ( f_var[217] )
 min_sales_constr_217: Sold_0[217] = MIN ( Q0_var[217] , 1046.30312518055 )
 max_lost_constr_217: Lost_0[217] = MAX ( Lost_0_expr_217 , 0 )
 max_Q1_constr_217: Q1_var[217] = MAX ( Q_hat_adjusted_217 , 0 )
 min_sales2_constr_217: Sold_1[217] = MIN ( total_stock_second_period_217
   , 246.9387680000446 )
 max_lost2_constr_217: Lost_1[217] = MAX ( Lost_1_expr_217 , 0 )
 GC1308: Fraction[218] = LOGISTIC ( f_var[218] )
 min_sales_constr_218: Sold_0[218] = MIN ( Q0_var[218] , 637.651619978532 )
 max_lost_constr_218: Lost_0[218] = MAX ( Lost_0_expr_218 , 0 )
 max_Q1_constr_218: Q1_var[218] = MAX ( Q_hat_adjusted_218 , 0 )
 min_sales2_constr_218: Sold_1[218] = MIN ( total_stock_second_period_218
   , 125.6262817679998 )
 max_lost2_constr_218: Lost_1[218] = MAX ( Lost_1_expr_218 , 0 )
 GC1314: Fraction[219] = LOGISTIC ( f_var[219] )
 min_sales_constr_219: Sold_0[219] = MIN ( Q0_var[219] , 1308.078874664296 )
 max_lost_constr_219: Lost_0[219] = MAX ( Lost_0_expr_219 , 0 )
 max_Q1_constr_219: Q1_var[219] = MAX ( Q_hat_adjusted_219 , 0 )
 min_sales2_constr_219: Sold_1[219] = MIN ( total_stock_second_period_219
   , 335.1713129221378 )
 max_lost2_constr_219: Lost_1[219] = MAX ( Lost_1_expr_219 , 0 )
 GC1320: Fraction[220] = LOGISTIC ( f_var[220] )
 min_sales_constr_220: Sold_0[220] = MIN ( Q0_var[220] , 790.9416768346563 )
 max_lost_constr_220: Lost_0[220] = MAX ( Lost_0_expr_220 , 0 )
 max_Q1_constr_220: Q1_var[220] = MAX ( Q_hat_adjusted_220 , 0 )
 min_sales2_constr_220: Sold_1[220] = MIN ( total_stock_second_period_220
   , 249.9526786238897 )
 max_lost2_constr_220: Lost_1[220] = MAX ( Lost_1_expr_220 , 0 )
 GC1326: Fraction[221] = LOGISTIC ( f_var[221] )
 min_sales_constr_221: Sold_0[221] = MIN ( Q0_var[221] , 1428.246646450981 )
 max_lost_constr_221: Lost_0[221] = MAX ( Lost_0_expr_221 , 0 )
 max_Q1_constr_221: Q1_var[221] = MAX ( Q_hat_adjusted_221 , 0 )
 min_sales2_constr_221: Sold_1[221] = MIN ( total_stock_second_period_221
   , 332.9768813596003 )
 max_lost2_constr_221: Lost_1[221] = MAX ( Lost_1_expr_221 , 0 )
 GC1332: Fraction[222] = LOGISTIC ( f_var[222] )
 min_sales_constr_222: Sold_0[222] = MIN ( Q0_var[222] , 640.7158418571609 )
 max_lost_constr_222: Lost_0[222] = MAX ( Lost_0_expr_222 , 0 )
 max_Q1_constr_222: Q1_var[222] = MAX ( Q_hat_adjusted_222 , 0 )
 min_sales2_constr_222: Sold_1[222] = MIN ( total_stock_second_period_222
   , 246.0132237408496 )
 max_lost2_constr_222: Lost_1[222] = MAX ( Lost_1_expr_222 , 0 )
 GC1338: Fraction[223] = LOGISTIC ( f_var[223] )
 min_sales_constr_223: Sold_0[223] = MIN ( Q0_var[223] , 2362.364538664486 )
 max_lost_constr_223: Lost_0[223] = MAX ( Lost_0_expr_223 , 0 )
 max_Q1_constr_223: Q1_var[223] = MAX ( Q_hat_adjusted_223 , 0 )
 min_sales2_constr_223: Sold_1[223] = MIN ( total_stock_second_period_223
   , 593.5073146967519 )
 max_lost2_constr_223: Lost_1[223] = MAX ( Lost_1_expr_223 , 0 )
 GC1344: Fraction[224] = LOGISTIC ( f_var[224] )
 min_sales_constr_224: Sold_0[224] = MIN ( Q0_var[224] , 1811.335975700625 )
 max_lost_constr_224: Lost_0[224] = MAX ( Lost_0_expr_224 , 0 )
 max_Q1_constr_224: Q1_var[224] = MAX ( Q_hat_adjusted_224 , 0 )
 min_sales2_constr_224: Sold_1[224] = MIN ( total_stock_second_period_224
   , 587.2680311326061 )
 max_lost2_constr_224: Lost_1[224] = MAX ( Lost_1_expr_224 , 0 )
 GC1350: Fraction[225] = LOGISTIC ( f_var[225] )
 min_sales_constr_225: Sold_0[225] = MIN ( Q0_var[225] , 1716.781094566551 )
 max_lost_constr_225: Lost_0[225] = MAX ( Lost_0_expr_225 , 0 )
 max_Q1_constr_225: Q1_var[225] = MAX ( Q_hat_adjusted_225 , 0 )
 min_sales2_constr_225: Sold_1[225] = MIN ( total_stock_second_period_225
   , 408.7676028306506 )
 max_lost2_constr_225: Lost_1[225] = MAX ( Lost_1_expr_225 , 0 )
 GC1356: Fraction[226] = LOGISTIC ( f_var[226] )
 min_sales_constr_226: Sold_0[226] = MIN ( Q0_var[226] , 2313.697709550831 )
 max_lost_constr_226: Lost_0[226] = MAX ( Lost_0_expr_226 , 0 )
 max_Q1_constr_226: Q1_var[226] = MAX ( Q_hat_adjusted_226 , 0 )
 min_sales2_constr_226: Sold_1[226] = MIN ( total_stock_second_period_226
   , 553.2800586397474 )
 max_lost2_constr_226: Lost_1[226] = MAX ( Lost_1_expr_226 , 0 )
 GC1362: Fraction[227] = LOGISTIC ( f_var[227] )
 min_sales_constr_227: Sold_0[227] = MIN ( Q0_var[227] , 1057.088594689651 )
 max_lost_constr_227: Lost_0[227] = MAX ( Lost_0_expr_227 , 0 )
 max_Q1_constr_227: Q1_var[227] = MAX ( Q_hat_adjusted_227 , 0 )
 min_sales2_constr_227: Sold_1[227] = MIN ( total_stock_second_period_227
   , 256.4679829320679 )
 max_lost2_constr_227: Lost_1[227] = MAX ( Lost_1_expr_227 , 0 )
 GC1368: Fraction[228] = LOGISTIC ( f_var[228] )
 min_sales_constr_228: Sold_0[228] = MIN ( Q0_var[228] , 1323.518782914005 )
 max_lost_constr_228: Lost_0[228] = MAX ( Lost_0_expr_228 , 0 )
 max_Q1_constr_228: Q1_var[228] = MAX ( Q_hat_adjusted_228 , 0 )
 min_sales2_constr_228: Sold_1[228] = MIN ( total_stock_second_period_228
   , 493.9464466248087 )
 max_lost2_constr_228: Lost_1[228] = MAX ( Lost_1_expr_228 , 0 )
 GC1374: Fraction[229] = LOGISTIC ( f_var[229] )
 min_sales_constr_229: Sold_0[229] = MIN ( Q0_var[229] , 2030.42417283989 )
 max_lost_constr_229: Lost_0[229] = MAX ( Lost_0_expr_229 , 0 )
 max_Q1_constr_229: Q1_var[229] = MAX ( Q_hat_adjusted_229 , 0 )
 min_sales2_constr_229: Sold_1[229] = MIN ( total_stock_second_period_229
   , 482.5262658576931 )
 max_lost2_constr_229: Lost_1[229] = MAX ( Lost_1_expr_229 , 0 )
 GC1380: Fraction[230] = LOGISTIC ( f_var[230] )
 min_sales_constr_230: Sold_0[230] = MIN ( Q0_var[230] , 1203.495015483821 )
 max_lost_constr_230: Lost_0[230] = MAX ( Lost_0_expr_230 , 0 )
 max_Q1_constr_230: Q1_var[230] = MAX ( Q_hat_adjusted_230 , 0 )
 min_sales2_constr_230: Sold_1[230] = MIN ( total_stock_second_period_230
   , 277.6572568116756 )
 max_lost2_constr_230: Lost_1[230] = MAX ( Lost_1_expr_230 , 0 )
 GC1386: Fraction[231] = LOGISTIC ( f_var[231] )
 min_sales_constr_231: Sold_0[231] = MIN ( Q0_var[231] , 1789.280656526285 )
 max_lost_constr_231: Lost_0[231] = MAX ( Lost_0_expr_231 , 0 )
 max_Q1_constr_231: Q1_var[231] = MAX ( Q_hat_adjusted_231 , 0 )
 min_sales2_constr_231: Sold_1[231] = MIN ( total_stock_second_period_231
   , 460.7358652304511 )
 max_lost2_constr_231: Lost_1[231] = MAX ( Lost_1_expr_231 , 0 )
 GC1392: Fraction[232] = LOGISTIC ( f_var[232] )
 min_sales_constr_232: Sold_0[232] = MIN ( Q0_var[232] , 450.4858931973076 )
 max_lost_constr_232: Lost_0[232] = MAX ( Lost_0_expr_232 , 0 )
 max_Q1_constr_232: Q1_var[232] = MAX ( Q_hat_adjusted_232 , 0 )
 min_sales2_constr_232: Sold_1[232] = MIN ( total_stock_second_period_232
   , 146.8087135993258 )
 max_lost2_constr_232: Lost_1[232] = MAX ( Lost_1_expr_232 , 0 )
 GC1398: Fraction[233] = LOGISTIC ( f_var[233] )
 min_sales_constr_233: Sold_0[233] = MIN ( Q0_var[233] , 929.9619383113279 )
 max_lost_constr_233: Lost_0[233] = MAX ( Lost_0_expr_233 , 0 )
 max_Q1_constr_233: Q1_var[233] = MAX ( Q_hat_adjusted_233 , 0 )
 min_sales2_constr_233: Sold_1[233] = MIN ( total_stock_second_period_233
   , 216.290593056255 )
 max_lost2_constr_233: Lost_1[233] = MAX ( Lost_1_expr_233 , 0 )
 GC1404: Fraction[234] = LOGISTIC ( f_var[234] )
 min_sales_constr_234: Sold_0[234] = MIN ( Q0_var[234] , 790.4798372181082 )
 max_lost_constr_234: Lost_0[234] = MAX ( Lost_0_expr_234 , 0 )
 max_Q1_constr_234: Q1_var[234] = MAX ( Q_hat_adjusted_234 , 0 )
 min_sales2_constr_234: Sold_1[234] = MIN ( total_stock_second_period_234
   , 204.9508814464238 )
 max_lost2_constr_234: Lost_1[234] = MAX ( Lost_1_expr_234 , 0 )
 GC1410: Fraction[235] = LOGISTIC ( f_var[235] )
 min_sales_constr_235: Sold_0[235] = MIN ( Q0_var[235] , 1596.6195067706 )
 max_lost_constr_235: Lost_0[235] = MAX ( Lost_0_expr_235 , 0 )
 max_Q1_constr_235: Q1_var[235] = MAX ( Q_hat_adjusted_235 , 0 )
 min_sales2_constr_235: Sold_1[235] = MIN ( total_stock_second_period_235
   , 412.8856690027548 )
 max_lost2_constr_235: Lost_1[235] = MAX ( Lost_1_expr_235 , 0 )
 GC1416: Fraction[236] = LOGISTIC ( f_var[236] )
 min_sales_constr_236: Sold_0[236] = MIN ( Q0_var[236] , 974.7963502925447 )
 max_lost_constr_236: Lost_0[236] = MAX ( Lost_0_expr_236 , 0 )
 max_Q1_constr_236: Q1_var[236] = MAX ( Q_hat_adjusted_236 , 0 )
 min_sales2_constr_236: Sold_1[236] = MIN ( total_stock_second_period_236
   , 200.8810863635611 )
 max_lost2_constr_236: Lost_1[236] = MAX ( Lost_1_expr_236 , 0 )
 GC1422: Fraction[237] = LOGISTIC ( f_var[237] )
 min_sales_constr_237: Sold_0[237] = MIN ( Q0_var[237] , 1044.087969164673 )
 max_lost_constr_237: Lost_0[237] = MAX ( Lost_0_expr_237 , 0 )
 max_Q1_constr_237: Q1_var[237] = MAX ( Q_hat_adjusted_237 , 0 )
 min_sales2_constr_237: Sold_1[237] = MIN ( total_stock_second_period_237
   , 241.7786000737533 )
 max_lost2_constr_237: Lost_1[237] = MAX ( Lost_1_expr_237 , 0 )
 GC1428: Fraction[238] = LOGISTIC ( f_var[238] )
 min_sales_constr_238: Sold_0[238] = MIN ( Q0_var[238] , 950.4955868498531 )
 max_lost_constr_238: Lost_0[238] = MAX ( Lost_0_expr_238 , 0 )
 max_Q1_constr_238: Q1_var[238] = MAX ( Q_hat_adjusted_238 , 0 )
 min_sales2_constr_238: Sold_1[238] = MIN ( total_stock_second_period_238
   , 236.1796951734634 )
 max_lost2_constr_238: Lost_1[238] = MAX ( Lost_1_expr_238 , 0 )
 GC1434: Fraction[239] = LOGISTIC ( f_var[239] )
 min_sales_constr_239: Sold_0[239] = MIN ( Q0_var[239] , 2116.922775514241 )
 max_lost_constr_239: Lost_0[239] = MAX ( Lost_0_expr_239 , 0 )
 max_Q1_constr_239: Q1_var[239] = MAX ( Q_hat_adjusted_239 , 0 )
 min_sales2_constr_239: Sold_1[239] = MIN ( total_stock_second_period_239
   , 571.6747450193398 )
 max_lost2_constr_239: Lost_1[239] = MAX ( Lost_1_expr_239 , 0 )
 GC1440: Fraction[240] = LOGISTIC ( f_var[240] )
 min_sales_constr_240: Sold_0[240] = MIN ( Q0_var[240] , 2566.210670256047 )
 max_lost_constr_240: Lost_0[240] = MAX ( Lost_0_expr_240 , 0 )
 max_Q1_constr_240: Q1_var[240] = MAX ( Q_hat_adjusted_240 , 0 )
 min_sales2_constr_240: Sold_1[240] = MIN ( total_stock_second_period_240
   , 573.999866948554 )
 max_lost2_constr_240: Lost_1[240] = MAX ( Lost_1_expr_240 , 0 )
 GC1446: Fraction[241] = LOGISTIC ( f_var[241] )
 min_sales_constr_241: Sold_0[241] = MIN ( Q0_var[241] , 1873.832010661456 )
 max_lost_constr_241: Lost_0[241] = MAX ( Lost_0_expr_241 , 0 )
 max_Q1_constr_241: Q1_var[241] = MAX ( Q_hat_adjusted_241 , 0 )
 min_sales2_constr_241: Sold_1[241] = MIN ( total_stock_second_period_241
   , 515.823297965149 )
 max_lost2_constr_241: Lost_1[241] = MAX ( Lost_1_expr_241 , 0 )
 GC1452: Fraction[242] = LOGISTIC ( f_var[242] )
 min_sales_constr_242: Sold_0[242] = MIN ( Q0_var[242] , 1684.619917154917 )
 max_lost_constr_242: Lost_0[242] = MAX ( Lost_0_expr_242 , 0 )
 max_Q1_constr_242: Q1_var[242] = MAX ( Q_hat_adjusted_242 , 0 )
 min_sales2_constr_242: Sold_1[242] = MIN ( total_stock_second_period_242
   , 325.7609810833633 )
 max_lost2_constr_242: Lost_1[242] = MAX ( Lost_1_expr_242 , 0 )
 GC1458: Fraction[243] = LOGISTIC ( f_var[243] )
 min_sales_constr_243: Sold_0[243] = MIN ( Q0_var[243] , 1898.759035566563 )
 max_lost_constr_243: Lost_0[243] = MAX ( Lost_0_expr_243 , 0 )
 max_Q1_constr_243: Q1_var[243] = MAX ( Q_hat_adjusted_243 , 0 )
 min_sales2_constr_243: Sold_1[243] = MIN ( total_stock_second_period_243
   , 345.0329836464568 )
 max_lost2_constr_243: Lost_1[243] = MAX ( Lost_1_expr_243 , 0 )
 GC1464: Fraction[244] = LOGISTIC ( f_var[244] )
 min_sales_constr_244: Sold_0[244] = MIN ( Q0_var[244] , 1175.431465401162 )
 max_lost_constr_244: Lost_0[244] = MAX ( Lost_0_expr_244 , 0 )
 max_Q1_constr_244: Q1_var[244] = MAX ( Q_hat_adjusted_244 , 0 )
 min_sales2_constr_244: Sold_1[244] = MIN ( total_stock_second_period_244
   , 287.288648000149 )
 max_lost2_constr_244: Lost_1[244] = MAX ( Lost_1_expr_244 , 0 )
 GC1470: Fraction[245] = LOGISTIC ( f_var[245] )
 min_sales_constr_245: Sold_0[245] = MIN ( Q0_var[245] , 1118.279925248586 )
 max_lost_constr_245: Lost_0[245] = MAX ( Lost_0_expr_245 , 0 )
 max_Q1_constr_245: Q1_var[245] = MAX ( Q_hat_adjusted_245 , 0 )
 min_sales2_constr_245: Sold_1[245] = MIN ( total_stock_second_period_245
   , 330.4147686996303 )
 max_lost2_constr_245: Lost_1[245] = MAX ( Lost_1_expr_245 , 0 )
 GC1476: Fraction[246] = LOGISTIC ( f_var[246] )
 min_sales_constr_246: Sold_0[246] = MIN ( Q0_var[246] , 672.0829463123032 )
 max_lost_constr_246: Lost_0[246] = MAX ( Lost_0_expr_246 , 0 )
 max_Q1_constr_246: Q1_var[246] = MAX ( Q_hat_adjusted_246 , 0 )
 min_sales2_constr_246: Sold_1[246] = MIN ( total_stock_second_period_246
   , 170.7519247014437 )
 max_lost2_constr_246: Lost_1[246] = MAX ( Lost_1_expr_246 , 0 )
 GC1482: Fraction[247] = LOGISTIC ( f_var[247] )
 min_sales_constr_247: Sold_0[247] = MIN ( Q0_var[247] , 1200.708040540061 )
 max_lost_constr_247: Lost_0[247] = MAX ( Lost_0_expr_247 , 0 )
 max_Q1_constr_247: Q1_var[247] = MAX ( Q_hat_adjusted_247 , 0 )
 min_sales2_constr_247: Sold_1[247] = MIN ( total_stock_second_period_247
   , 235.4922312769147 )
 max_lost2_constr_247: Lost_1[247] = MAX ( Lost_1_expr_247 , 0 )
 GC1488: Fraction[248] = LOGISTIC ( f_var[248] )
 min_sales_constr_248: Sold_0[248] = MIN ( Q0_var[248] , 870.7986115906817 )
 max_lost_constr_248: Lost_0[248] = MAX ( Lost_0_expr_248 , 0 )
 max_Q1_constr_248: Q1_var[248] = MAX ( Q_hat_adjusted_248 , 0 )
 min_sales2_constr_248: Sold_1[248] = MIN ( total_stock_second_period_248
   , 192.5506531895341 )
 max_lost2_constr_248: Lost_1[248] = MAX ( Lost_1_expr_248 , 0 )
 GC1494: Fraction[249] = LOGISTIC ( f_var[249] )
 min_sales_constr_249: Sold_0[249] = MIN ( Q0_var[249] , 2022.246963987618 )
 max_lost_constr_249: Lost_0[249] = MAX ( Lost_0_expr_249 , 0 )
 max_Q1_constr_249: Q1_var[249] = MAX ( Q_hat_adjusted_249 , 0 )
 min_sales2_constr_249: Sold_1[249] = MIN ( total_stock_second_period_249
   , 540.4530776953865 )
 max_lost2_constr_249: Lost_1[249] = MAX ( Lost_1_expr_249 , 0 )
End
