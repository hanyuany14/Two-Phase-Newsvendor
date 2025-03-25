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
 R2: - 0.1994354498475496 alphas[0] + 0.5959541390893992 alphas[1]
   - 0.3160290049521312 alphas[2] + 0.2282012730856178 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 2156.17943612574 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 311.0251636130733
 R5: Q0_var[0] + Q_hat_adjusted_0 = 2020.869748212403
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 1562.819790626047
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.7880526840920714 alphas[0] + 0.3824338004940984 alphas[1]
   + 1.594375883979111 alphas[2] - 1.083380785319452 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2156.17943612574 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 105.9702103844509
 R15: Q0_var[1] + Q_hat_adjusted_1 = 672.817955299721
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 324.5138204310281
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.390303428307552 alphas[0] - 0.1353514409021558 alphas[1]
   + 0.0676606266545596 alphas[2] + 0.9843978873682864 alphas[3]
   - alphas[4] + f_var[2] = 0
 R23: - 2156.17943612574 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 534.4887839207412
 R25: Q0_var[2] + Q_hat_adjusted_2 = 1920.915405140254
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 974.5514912467027
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.1855304602549332 alphas[0] - 0.9042633654178782 alphas[1]
   - 0.6959033283355666 alphas[2] + 0.0697079345154207 alphas[3]
   - alphas[4] + f_var[3] = 0
 R33: - 2156.17943612574 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 459.6870818321154
 R35: Q0_var[3] + Q_hat_adjusted_3 = 2357.167032778966
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 1886.664143052687
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.2433605750913124 alphas[0] + 0.5171402568792384 alphas[1]
   + 1.483217159751566 alphas[2] - 1.394735561647883 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2156.17943612574 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 274.6221011453696
 R45: Q0_var[4] + Q_hat_adjusted_4 = 858.4413559826968
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 825.6712004476807
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.5428946440514224 alphas[0] - 1.495755798507852 alphas[1]
   - 1.25560553144313 alphas[2] - 0.5127083249705495 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2156.17943612574 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 622.1284308325011
 R55: Q0_var[5] + Q_hat_adjusted_5 = 2816.351183706361
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 2092.033355181434
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.1940694240047668 alphas[0] - 1.285008892009203 alphas[1]
   - 0.0774028354093683 alphas[2] - 0.9525868099842801 alphas[3]
   - alphas[4] + f_var[6] = 0
 R63: - 2156.17943612574 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 417.1972146883232
 R65: Q0_var[6] + Q_hat_adjusted_6 = 1937.65622554416
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 1471.450587105507
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.412783753680732 alphas[0] + 1.487776157999748 alphas[1]
   + 1.526860159618238 alphas[2] - 0.1220517709924216 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2156.17943612574 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 57.70010585339908
 R75: Q0_var[7] + Q_hat_adjusted_7 = 689.8725141631603
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 379.7910369710239
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.667120836682995 alphas[0] + 0.7893140852491326 alphas[1]
   + 0.8917618104409608 alphas[2] - 1.565651659901687 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2156.17943612574 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 304.7706936787965
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1250.826961454708
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 619.0432990857677
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.3856301904530194 alphas[0] + 0.133312182771857 alphas[1]
   - 1.498674410859705 alphas[2] - 0.2486670685731852 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2156.17943612574 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 655.0289795422884
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2991.613584385294
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 2150.174818704398
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: - 1.058826429400216 alphas[0] + 1.297144968886463 alphas[1]
   - 0.281412791603891 alphas[2] + 1.577045591015342 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 2156.17943612574 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 276.3227474035778
 R105: Q0_var[10] + Q_hat_adjusted_10 = 1976.650870625119
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 2075.190483876526
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: - 0.1289659613381638 alphas[0] + 0.4661286340369206 alphas[1]
   + 1.275080940335017 alphas[2] + 1.064536193573597 alphas[3] - alphas[4]
   + f_var[11] = 0
 R113: - 2156.17943612574 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 119.5458267768758
 R115: Q0_var[11] + Q_hat_adjusted_11 = 886.1855888482737
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 928.2942621173393
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: - 0.2674725527941534 alphas[0] - 0.8684886690789593 alphas[1]
   + 0.6848261007626395 alphas[2] + 0.3694434328435542 alphas[3]
   - alphas[4] + f_var[12] = 0
 R123: - 2156.17943612574 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 175.550333593017
 R125: Q0_var[12] + Q_hat_adjusted_12 = 1299.199509139134
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 1120.281080114851
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 1.532447601975659 alphas[0] - 0.66543404950399 alphas[1]
   + 0.1303421850897499 alphas[2] - 0.2910703398163376 alphas[3]
   - alphas[4] + f_var[13] = 0
 R133: - 2156.17943612574 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 421.8738478845578
 R135: Q0_var[13] + Q_hat_adjusted_13 = 1808.9048108712
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 1319.880339667429
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: 1.490882091150807 alphas[0] - 0.4673311191168003 alphas[1]
   + 1.558691618537723 alphas[2] - 0.8646941723404371 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 2156.17943612574 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 290.0192850593794
 R145: Q0_var[14] + Q_hat_adjusted_14 = 816.61085807823
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 308.4142764821366
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 1.433946192128581 alphas[0] - 0.7677192798408546 alphas[1]
   + 0.2781357684317148 alphas[2] + 0.3415023656370222 alphas[3]
   - alphas[4] + f_var[15] = 0
 R153: - 2156.17943612574 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 383.6992953883416
 R155: Q0_var[15] + Q_hat_adjusted_15 = 1690.683452658845
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 1158.411768917133
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 1.670668692922823 alphas[0] + 0.9553442883159851 alphas[1]
   + 0.0229750766403372 alphas[2] - 0.6567199095951021 alphas[3]
   - alphas[4] + f_var[16] = 0
 R163: - 2156.17943612574 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 354.6680610536623
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1833.87652167819
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 1164.255632456589
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 1.203507202869476 alphas[0] + 0.4892271291372563 alphas[1]
   + 0.8286157976539109 alphas[2] - 0.9249918938197198 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 2156.17943612574 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 110.9118922780958
 R175: Q0_var[17] + Q_hat_adjusted_17 = 1166.484213238534
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 862.3794294522688
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 1.010823506811682 alphas[0] - 0.6508753925559209 alphas[1]
   - 0.3021266140051108 alphas[2] + 0.6636051862218381 alphas[3]
   - alphas[4] + f_var[18] = 0
 R183: - 2156.17943612574 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 392.2528102253823
 R185: Q0_var[18] + Q_hat_adjusted_18 = 2064.170016620597
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 1666.377769428032
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.335796547448608 alphas[0] - 1.342658108736613 alphas[1]
   - 0.8416699222357673 alphas[2] - 1.561630770602946 alphas[3] - alphas[4]
   + f_var[19] = 0
 R193: - 2156.17943612574 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 571.6462479792019
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2521.445396283552
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 1729.71322047711
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 1.720031836067636 alphas[0] - 0.1991056009112765 alphas[1]
   + 0.8543263731462359 alphas[2] - 0.4536893355968431 alphas[3]
   - alphas[4] + f_var[20] = 0
 R203: - 2156.17943612574 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 203.8340100004
 R205: Q0_var[20] + Q_hat_adjusted_20 = 1210.055018295968
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 712.982219600498
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: - 1.085125325943327 alphas[0] + 0.6902895413084772 alphas[1]
   - 0.4065629353121455 alphas[2] - 1.231305288074461 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 2156.17943612574 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 278.7784712176002
 R215: Q0_var[21] + Q_hat_adjusted_21 = 2057.596330326662
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 1624.237131575594
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.1095419473062271 alphas[0] + 1.550669565800169 alphas[1]
   - 0.4692437486418328 alphas[2] - 0.1386692537035579 alphas[3]
   - alphas[4] + f_var[22] = 0
 R223: - 2156.17943612574 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 476.0153771204417
 R225: Q0_var[22] + Q_hat_adjusted_22 = 2223.896934589831
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 1767.007096298152
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 1.019216854377619 alphas[0] - 0.767066410641397 alphas[1]
   - 1.481841182870796 alphas[2] - 0.7627524822581606 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2156.17943612574 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 672.6244800796422
 R235: Q0_var[23] + Q_hat_adjusted_23 = 2992.227434570736
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 2348.230435223977
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 1.323758585853078 alphas[0] + 1.626840651275249 alphas[1]
   - 0.9237351722219811 alphas[2] - 0.8600326567812047 alphas[3]
   - alphas[4] + f_var[24] = 0
 R243: - 2156.17943612574 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 518.4956700972725
 R245: Q0_var[24] + Q_hat_adjusted_24 = 2539.388974497249
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 1959.719550616583
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: - 0.5217625838612794 alphas[0] + 0.3836463203994959 alphas[1]
   - 1.153413533613816 alphas[2] - 1.322902672596358 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 2156.17943612574 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 473.3862630798172
 R255: Q0_var[25] + Q_hat_adjusted_25 = 2656.107376685296
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 2131.774750330777
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: 1.235032673928118 alphas[0] - 0.1543225902170167 alphas[1]
   + 0.0198241356407334 alphas[2] + 1.22464669540647 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 2156.17943612574 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 401.99349531143
 R265: Q0_var[26] + Q_hat_adjusted_26 = 1866.240284735306
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 1485.560539390864
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.599922966536701 alphas[0] - 1.488778734615604 alphas[1]
   + 1.019812414729911 alphas[2] - 0.1465540505077418 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 2156.17943612574 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 96.42274920631569
 R275: Q0_var[27] + Q_hat_adjusted_27 = 1036.706666541056
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 939.7428894940106
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: - 0.1040359686919687 alphas[0] + 1.348718287721178 alphas[1]
   - 1.636803102919747 alphas[2] + 0.2900995655762993 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 2156.17943612574 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 707.4595453920876
 R285: Q0_var[28] + Q_hat_adjusted_28 = 3112.854012745563
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 2407.558964246491
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.2751761997847126 alphas[0] + 0.2702876112210127 alphas[1]
   + 1.203350380168186 alphas[2] - 1.373022299849649 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 2156.17943612574 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 82.32632633032061
 R295: Q0_var[29] + Q_hat_adjusted_29 = 910.4466683483479
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 462.9703976355289
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 0.8062338319695235 alphas[0] + 1.570851503547021 alphas[1]
   + 0.081981295260451 alphas[2] + 1.376764013977475 alphas[3] - alphas[4]
   + f_var[30] = 0
 R303: - 2156.17943612574 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 487.8714369750719
 R305: Q0_var[30] + Q_hat_adjusted_30 = 1881.919792419082
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 1068.460843962106
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 700 Sold_0[30] - 700 Sold_1[30] + 700 Lost_0[30] + 700 Lost_1[30]
   + 300 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: - 0.99694420013121 alphas[0] + 0.4861161373109953 alphas[1]
   + 1.404162254092736 alphas[2] - 1.572735432584875 alphas[3] - alphas[4]
   + f_var[31] = 0
 R313: - 2156.17943612574 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 214.6246109264699
 R315: Q0_var[31] + Q_hat_adjusted_31 = 868.259767982584
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 582.6870558914214
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 700 Sold_0[31] - 700 Sold_1[31] + 700 Lost_0[31] + 700 Lost_1[31]
   + 300 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: 0.1283953968097999 alphas[0] - 0.4668084464053881 alphas[1]
   - 1.466469732053739 alphas[2] - 1.21266086105037 alphas[3] - alphas[4]
   + f_var[32] = 0
 R323: - 2156.17943612574 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 541.7519399273892
 R325: Q0_var[32] + Q_hat_adjusted_32 = 2898.510116716918
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 2291.582259291944
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 700 Sold_0[32] - 700 Sold_1[32] + 700 Lost_0[32] + 700 Lost_1[32]
   + 300 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: - 0.2688501586056804 alphas[0] + 0.7026661162233164 alphas[1]
   - 1.535548124573649 alphas[2] + 1.481867549757902 alphas[3] - alphas[4]
   + f_var[33] = 0
 R333: - 2156.17943612574 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 366.5223049134499
 R335: Q0_var[33] + Q_hat_adjusted_33 = 2829.890635432949
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 2488.059168933316
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 700 Sold_0[33] - 700 Sold_1[33] + 700 Lost_0[33] + 700 Lost_1[33]
   + 300 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: 1.675722892522268 alphas[0] + 0.9388967411265299 alphas[1]
   - 0.7307100959849104 alphas[2] + 0.079720907849908 alphas[3] - alphas[4]
   + f_var[34] = 0
 R343: - 2156.17943612574 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 375.6855613206947
 R345: Q0_var[34] + Q_hat_adjusted_34 = 2325.345298182239
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 2049.738536111399
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 700 Sold_0[34] - 700 Sold_1[34] + 700 Lost_0[34] + 700 Lost_1[34]
   + 300 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: - 0.4429191539555587 alphas[0] + 1.259401011673255 alphas[1]
   + 0.4801277072855211 alphas[2] - 0.6819339217586057 alphas[3]
   - alphas[4] + f_var[35] = 0
 R353: - 2156.17943612574 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 185.5582342339428
 R355: Q0_var[35] + Q_hat_adjusted_35 = 1435.455897233579
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 1017.892244713687
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 700 Sold_0[35] - 700 Sold_1[35] + 700 Lost_0[35] + 700 Lost_1[35]
   + 300 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 0.4233201171100196 alphas[0] - 0.3131422762694678 alphas[1]
   + 0.7935276080512971 alphas[2] - 0.751150636101142 alphas[3] - alphas[4]
   + f_var[36] = 0
 R363: - 2156.17943612574 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 130.7781133225293
 R365: Q0_var[36] + Q_hat_adjusted_36 = 1201.480893382769
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 856.1232990882058
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 700 Sold_0[36] - 700 Sold_1[36] + 700 Lost_0[36] + 700 Lost_1[36]
   + 300 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.4404373329621946 alphas[0] - 0.7157197892401143 alphas[1]
   + 0.7577300923618497 alphas[2] - 1.317006041399737 alphas[3] - alphas[4]
   + f_var[37] = 0
 R373: - 2156.17943612574 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 134.2424139460701
 R375: Q0_var[37] + Q_hat_adjusted_37 = 1226.444440600107
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 920.3682126844783
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 700 Sold_0[37] - 700 Sold_1[37] + 700 Lost_0[37] + 700 Lost_1[37]
   + 300 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: - 1.596665153632425 alphas[0] - 0.6362712642744018 alphas[1]
   + 1.210108509524166 alphas[2] - 0.9585915477882951 alphas[3] - alphas[4]
   + f_var[38] = 0
 R383: - 2156.17943612574 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 405.8293392306349
 R385: Q0_var[38] + Q_hat_adjusted_38 = 1118.485116430218
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 819.673390786543
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 700 Sold_0[38] - 700 Sold_1[38] + 700 Lost_0[38] + 700 Lost_1[38]
   + 300 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 0.6699970491189249 alphas[0] - 1.579939257964751 alphas[1]
   - 0.0961770331338674 alphas[2] + 0.6848818063163583 alphas[3]
   - alphas[4] + f_var[39] = 0
 R393: - 2156.17943612574 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 299.5267134285999
 R395: Q0_var[39] + Q_hat_adjusted_39 = 1874.064732540635
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 1542.059211743204
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 700 Sold_0[39] - 700 Sold_1[39] + 700 Lost_0[39] + 700 Lost_1[39]
   + 300 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: 0.4703043731975303 alphas[0] + 1.644496740840545 alphas[1]
   + 1.162502985719682 alphas[2] + 0.0546171301692974 alphas[3] - alphas[4]
   + f_var[40] = 0
 R403: - 2156.17943612574 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 103.363624857212
 R405: Q0_var[40] + Q_hat_adjusted_40 = 950.0321763618971
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 735.0206445377808
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 700 Sold_0[40] - 700 Sold_1[40] + 700 Lost_0[40] + 700 Lost_1[40]
   + 300 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: 0.1960338562701699 alphas[0] - 0.5518893524061027 alphas[1]
   + 0.5891572517699288 alphas[2] - 0.628913143986337 alphas[3] - alphas[4]
   + f_var[41] = 0
 R413: - 2156.17943612574 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 440.4359615854938
 R415: Q0_var[41] + Q_hat_adjusted_41 = 1529.837600069321
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 806.7483530363293
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 700 Sold_0[41] - 700 Sold_1[41] + 700 Lost_0[41] + 700 Lost_1[41]
   + 300 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.7259340475667077 alphas[0] - 0.3923833531611142 alphas[1]
   - 0.4269073678951803 alphas[2] + 1.264689966993831 alphas[3] - alphas[4]
   + f_var[42] = 0
 R423: - 2156.17943612574 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 478.1457423538777
 R425: Q0_var[42] + Q_hat_adjusted_42 = 2198.414167088618
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 1963.242512356521
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 700 Sold_0[42] - 700 Sold_1[42] + 700 Lost_0[42] + 700 Lost_1[42]
   + 300 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: 1.529128610045219 alphas[0] - 0.3526720596337787 alphas[1]
   - 0.7574397537223503 alphas[2] + 1.077954641288407 alphas[3] - alphas[4]
   + f_var[43] = 0
 R433: - 2156.17943612574 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 390.4136837566275
 R435: Q0_var[43] + Q_hat_adjusted_43 = 2351.746801728822
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 1870.743024037578
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 700 Sold_0[43] - 700 Sold_1[43] + 700 Lost_0[43] + 700 Lost_1[43]
   + 300 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: - 0.6167393310346315 alphas[0] - 1.627341123146449 alphas[1]
   + 1.547800406626519 alphas[2] + 0.113078562532797 alphas[3] - alphas[4]
   + f_var[44] = 0
 R443: - 2156.17943612574 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 138.5934337217769
 R445: Q0_var[44] + Q_hat_adjusted_44 = 740.337543913573
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 532.6260551433593
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 700 Sold_0[44] - 700 Sold_1[44] + 700 Lost_0[44] + 700 Lost_1[44]
   + 300 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: - 0.63043499595176 alphas[0] + 1.596211676283536 alphas[1]
   + 0.2174933788676494 alphas[2] + 0.7275189044495836 alphas[3]
   - alphas[4] + f_var[45] = 0
 R453: - 2156.17943612574 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 401.9114084672532
 R455: Q0_var[45] + Q_hat_adjusted_45 = 1740.824581196658
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 1033.406295765255
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 700 Sold_0[45] - 700 Sold_1[45] + 700 Lost_0[45] + 700 Lost_1[45]
   + 300 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: 0.9978913897408717 alphas[0] - 0.7198516434956329 alphas[1]
   - 0.5102250027293713 alphas[2] - 1.16932631103332 alphas[3] - alphas[4]
   + f_var[46] = 0
 R463: - 2156.17943612574 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 451.0708956474153
 R465: Q0_var[46] + Q_hat_adjusted_46 = 2233.882442850864
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 1824.456664841893
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 700 Sold_0[46] - 700 Sold_1[46] + 700 Lost_0[46] + 700 Lost_1[46]
   + 300 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: 1.286073580239811 alphas[0] - 1.119247711193984 alphas[1]
   - 1.169477478609894 alphas[2] - 0.2406508510574602 alphas[3] - alphas[4]
   + f_var[47] = 0
 R473: - 2156.17943612574 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 701.632369255298
 R475: Q0_var[47] + Q_hat_adjusted_47 = 2812.739508401006
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 1919.353389766453
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 700 Sold_0[47] - 700 Sold_1[47] + 700 Lost_0[47] + 700 Lost_1[47]
   + 300 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: 0.6262523708228174 alphas[0] - 0.0836762185196377 alphas[1]
   + 0.0693042022149277 alphas[2] + 0.5580425334677064 alphas[3]
   - alphas[4] + f_var[48] = 0
 R483: - 2156.17943612574 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 375.004082192915
 R485: Q0_var[48] + Q_hat_adjusted_48 = 1817.543997736497
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 1399.854312327068
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 700 Sold_0[48] - 700 Sold_1[48] + 700 Lost_0[48] + 700 Lost_1[48]
   + 300 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: 0.4554351373838961 alphas[0] + 0.5155814584569119 alphas[1]
   - 1.549580517629781 alphas[2] - 1.123187905840042 alphas[3] - alphas[4]
   + f_var[49] = 0
 R493: - 2156.17943612574 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 676.143265321571
 R495: Q0_var[49] + Q_hat_adjusted_49 = 3037.445093488773
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 2042.483713405973
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 700 Sold_0[49] - 700 Sold_1[49] + 700 Lost_0[49] + 700 Lost_1[49]
   + 300 Left_1[49] + profits_vars[49] = 0
 Fraction_lower_bound_50: Fraction[50] >= 0
 Fraction_upper_bound_50: Fraction[50] <= 1
 R502: - 0.2750868033300156 alphas[0] - 1.041415834296586 alphas[1]
   + 0.756650937677515 alphas[2] - 0.9754419518550075 alphas[3] - alphas[4]
   + f_var[50] = 0
 R503: - 2156.17943612574 Fraction[50] + Q0_var[50] = 0
 R504: Q0_var[50] + Lost_0_expr_50 = 130.7830838166814
 R505: Q0_var[50] + Q_hat_adjusted_50 = 1224.956625586383
 R506: Sold_0[50] - Q0_var[50] - Q1_var[50] + total_stock_second_period_50
   = 0
 R507: total_stock_second_period_50 + Lost_1_expr_50 = 836.2830875038119
 R508: Sold_1[50] + Left_1[50] - total_stock_second_period_50 = 0
 R509: - 700 Sold_0[50] - 700 Sold_1[50] + 700 Lost_0[50] + 700 Lost_1[50]
   + 300 Left_1[50] + profits_vars[50] = 0
 Fraction_lower_bound_51: Fraction[51] >= 0
 Fraction_upper_bound_51: Fraction[51] <= 1
 R512: 0.1904809482087055 alphas[0] + 1.322312378308099 alphas[1]
   + 1.583083539043155 alphas[2] - 1.625780583856278 alphas[3] - alphas[4]
   + f_var[51] = 0
 R513: - 2156.17943612574 Fraction[51] + Q0_var[51] = 0
 R514: Q0_var[51] + Lost_0_expr_51 = 104.2038153379804
 R515: Q0_var[51] + Q_hat_adjusted_51 = 683.2655250470441
 R516: Sold_0[51] - Q0_var[51] - Q1_var[51] + total_stock_second_period_51
   = 0
 R517: total_stock_second_period_51 + Lost_1_expr_51 = 597.627207980269
 R518: Sold_1[51] + Left_1[51] - total_stock_second_period_51 = 0
 R519: - 700 Sold_0[51] - 700 Sold_1[51] + 700 Lost_0[51] + 700 Lost_1[51]
   + 300 Left_1[51] + profits_vars[51] = 0
 Fraction_lower_bound_52: Fraction[52] >= 0
 Fraction_upper_bound_52: Fraction[52] <= 1
 R522: - 1.754545574344733 alphas[0] + 0.1472842540449275 alphas[1]
   + 0.0416779289074204 alphas[2] - 1.118060739903543 alphas[3] - alphas[4]
   + f_var[52] = 0
 R523: - 2156.17943612574 Fraction[52] + Q0_var[52] = 0
 R524: Q0_var[52] + Lost_0_expr_52 = 415.4612156881613
 R525: Q0_var[52] + Q_hat_adjusted_52 = 1861.022089008006
 R526: Sold_0[52] - Q0_var[52] - Q1_var[52] + total_stock_second_period_52
   = 0
 R527: total_stock_second_period_52 + Lost_1_expr_52 = 1610.044917623197
 R528: Sold_1[52] + Left_1[52] - total_stock_second_period_52 = 0
 R529: - 700 Sold_0[52] - 700 Sold_1[52] + 700 Lost_0[52] + 700 Lost_1[52]
   + 300 Left_1[52] + profits_vars[52] = 0
 Fraction_lower_bound_53: Fraction[53] >= 0
 Fraction_upper_bound_53: Fraction[53] <= 1
 R532: 1.381176957874878 alphas[0] - 0.1182502364665632 alphas[1]
   + 0.7878523003868081 alphas[2] + 0.3368959654818358 alphas[3]
   - alphas[4] + f_var[53] = 0
 R533: - 2156.17943612574 Fraction[53] + Q0_var[53] = 0
 R534: Q0_var[53] + Lost_0_expr_53 = 303.3280717677706
 R535: Q0_var[53] + Q_hat_adjusted_53 = 1315.958137375863
 R536: Sold_0[53] - Q0_var[53] - Q1_var[53] + total_stock_second_period_53
   = 0
 R537: total_stock_second_period_53 + Lost_1_expr_53 = 761.4476958833377
 R538: Sold_1[53] + Left_1[53] - total_stock_second_period_53 = 0
 R539: - 700 Sold_0[53] - 700 Sold_1[53] + 700 Lost_0[53] + 700 Lost_1[53]
   + 300 Left_1[53] + profits_vars[53] = 0
 Fraction_lower_bound_54: Fraction[54] >= 0
 Fraction_upper_bound_54: Fraction[54] <= 1
 R542: 1.00321874161168 alphas[0] - 0.7112431544980005 alphas[1]
   - 1.072687134975786 alphas[2] + 1.301456087461202 alphas[3] - alphas[4]
   + f_var[54] = 0
 R543: - 2156.17943612574 Fraction[54] + Q0_var[54] = 0
 R544: Q0_var[54] + Lost_0_expr_54 = 493.1986700210101
 R545: Q0_var[54] + Q_hat_adjusted_54 = 2617.62286586701
 R546: Sold_0[54] - Q0_var[54] - Q1_var[54] + total_stock_second_period_54
   = 0
 R547: total_stock_second_period_54 + Lost_1_expr_54 = 2247.345197135545
 R548: Sold_1[54] + Left_1[54] - total_stock_second_period_54 = 0
 R549: - 700 Sold_0[54] - 700 Sold_1[54] + 700 Lost_0[54] + 700 Lost_1[54]
   + 300 Left_1[54] + profits_vars[54] = 0
 Fraction_lower_bound_55: Fraction[55] >= 0
 Fraction_upper_bound_55: Fraction[55] <= 1
 R552: 1.171505750221268 alphas[0] + 1.343977225067654 alphas[1]
   + 0.8643617839666495 alphas[2] - 1.095651474318477 alphas[3] - alphas[4]
   + f_var[55] = 0
 R553: - 2156.17943612574 Fraction[55] + Q0_var[55] = 0
 R554: Q0_var[55] + Lost_0_expr_55 = 133.8092946906124
 R555: Q0_var[55] + Q_hat_adjusted_55 = 1158.548334816406
 R556: Sold_0[55] - Q0_var[55] - Q1_var[55] + total_stock_second_period_55
   = 0
 R557: total_stock_second_period_55 + Lost_1_expr_55 = 775.3455395011633
 R558: Sold_1[55] + Left_1[55] - total_stock_second_period_55 = 0
 R559: - 700 Sold_0[55] - 700 Sold_1[55] + 700 Lost_0[55] + 700 Lost_1[55]
   + 300 Left_1[55] + profits_vars[55] = 0
 Fraction_lower_bound_56: Fraction[56] >= 0
 Fraction_upper_bound_56: Fraction[56] <= 1
 R562: - 0.5684176357101539 alphas[0] + 0.8775215170440438 alphas[1]
   + 0.6091570889177848 alphas[2] + 1.529155999360939 alphas[3] - alphas[4]
   + f_var[56] = 0
 R563: - 2156.17943612574 Fraction[56] + Q0_var[56] = 0
 R564: Q0_var[56] + Lost_0_expr_56 = 127.655010513551
 R565: Q0_var[56] + Q_hat_adjusted_56 = 1316.450349807038
 R566: Sold_0[56] - Q0_var[56] - Q1_var[56] + total_stock_second_period_56
   = 0
 R567: total_stock_second_period_56 + Lost_1_expr_56 = 1204.369459325605
 R568: Sold_1[56] + Left_1[56] - total_stock_second_period_56 = 0
 R569: - 700 Sold_0[56] - 700 Sold_1[56] + 700 Lost_0[56] + 700 Lost_1[56]
   + 300 Left_1[56] + profits_vars[56] = 0
 Fraction_lower_bound_57: Fraction[57] >= 0
 Fraction_upper_bound_57: Fraction[57] <= 1
 R572: 0.84608450537325 alphas[0] + 0.2554316404627066 alphas[1]
   - 0.9661879070782511 alphas[2] + 0.5526442009094482 alphas[3]
   - alphas[4] + f_var[57] = 0
 R573: - 2156.17943612574 Fraction[57] + Q0_var[57] = 0
 R574: Q0_var[57] + Lost_0_expr_57 = 436.73253927975
 R575: Q0_var[57] + Q_hat_adjusted_57 = 2513.852445282177
 R576: Sold_0[57] - Q0_var[57] - Q1_var[57] + total_stock_second_period_57
   = 0
 R577: total_stock_second_period_57 + Lost_1_expr_57 = 2157.210000812454
 R578: Sold_1[57] + Left_1[57] - total_stock_second_period_57 = 0
 R579: - 700 Sold_0[57] - 700 Sold_1[57] + 700 Lost_0[57] + 700 Lost_1[57]
   + 300 Left_1[57] + profits_vars[57] = 0
 Fraction_lower_bound_58: Fraction[58] >= 0
 Fraction_upper_bound_58: Fraction[58] <= 1
 R582: 0.0924490138418171 alphas[0] - 0.4704331650592617 alphas[1]
   - 0.7167928251199288 alphas[2] - 1.361523045123415 alphas[3] - alphas[4]
   + f_var[58] = 0
 R583: - 2156.17943612574 Fraction[58] + Q0_var[58] = 0
 R584: Q0_var[58] + Lost_0_expr_58 = 643.537895569871
 R585: Q0_var[58] + Q_hat_adjusted_58 = 2488.375149992338
 R586: Sold_0[58] - Q0_var[58] - Q1_var[58] + total_stock_second_period_58
   = 0
 R587: total_stock_second_period_58 + Lost_1_expr_58 = 1907.556576283753
 R588: Sold_1[58] + Left_1[58] - total_stock_second_period_58 = 0
 R589: - 700 Sold_0[58] - 700 Sold_1[58] + 700 Lost_0[58] + 700 Lost_1[58]
   + 300 Left_1[58] + profits_vars[58] = 0
 Fraction_lower_bound_59: Fraction[59] >= 0
 Fraction_upper_bound_59: Fraction[59] <= 1
 R592: 0.8774513546569805 alphas[0] - 1.368987808694591 alphas[1]
   - 0.2018634854612618 alphas[2] - 1.499433456222205 alphas[3] - alphas[4]
   + f_var[59] = 0
 R593: - 2156.17943612574 Fraction[59] + Q0_var[59] = 0
 R594: Q0_var[59] + Lost_0_expr_59 = 353.7419563365844
 R595: Q0_var[59] + Q_hat_adjusted_59 = 1975.873875275218
 R596: Sold_0[59] - Q0_var[59] - Q1_var[59] + total_stock_second_period_59
   = 0
 R597: total_stock_second_period_59 + Lost_1_expr_59 = 1514.763625437006
 R598: Sold_1[59] + Left_1[59] - total_stock_second_period_59 = 0
 R599: - 700 Sold_0[59] - 700 Sold_1[59] + 700 Lost_0[59] + 700 Lost_1[59]
   + 300 Left_1[59] + profits_vars[59] = 0
 Fraction_lower_bound_60: Fraction[60] >= 0
 Fraction_upper_bound_60: Fraction[60] <= 1
 R602: 1.179784148302056 alphas[0] - 0.4552048304564248 alphas[1]
   - 0.6833741787620899 alphas[2] - 0.423128429713686 alphas[3] - alphas[4]
   + f_var[60] = 0
 R603: - 2156.17943612574 Fraction[60] + Q0_var[60] = 0
 R604: Q0_var[60] + Lost_0_expr_60 = 598.9061411100495
 R605: Q0_var[60] + Q_hat_adjusted_60 = 2438.545125013567
 R606: Sold_0[60] - Q0_var[60] - Q1_var[60] + total_stock_second_period_60
   = 0
 R607: total_stock_second_period_60 + Lost_1_expr_60 = 1834.376561371839
 R608: Sold_1[60] + Left_1[60] - total_stock_second_period_60 = 0
 R609: - 700 Sold_0[60] - 700 Sold_1[60] + 700 Lost_0[60] + 700 Lost_1[60]
   + 300 Left_1[60] + profits_vars[60] = 0
 Fraction_lower_bound_61: Fraction[61] >= 0
 Fraction_upper_bound_61: Fraction[61] <= 1
 R612: 1.35170527864262 alphas[0] + 1.198882248096083 alphas[1]
   + 0.2551206006141684 alphas[2] - 0.5847538182599105 alphas[3]
   - alphas[4] + f_var[61] = 0
 R613: - 2156.17943612574 Fraction[61] + Q0_var[61] = 0
 R614: Q0_var[61] + Lost_0_expr_61 = 377.1809117046377
 R615: Q0_var[61] + Q_hat_adjusted_61 = 1701.096065024885
 R616: Sold_0[61] - Q0_var[61] - Q1_var[61] + total_stock_second_period_61
   = 0
 R617: total_stock_second_period_61 + Lost_1_expr_61 = 1212.213327660204
 R618: Sold_1[61] + Left_1[61] - total_stock_second_period_61 = 0
 R619: - 700 Sold_0[61] - 700 Sold_1[61] + 700 Lost_0[61] + 700 Lost_1[61]
   + 300 Left_1[61] + profits_vars[61] = 0
 Fraction_lower_bound_62: Fraction[62] >= 0
 Fraction_upper_bound_62: Fraction[62] <= 1
 R622: - 0.5798140695348224 alphas[0] - 1.68803565167442 alphas[1]
   + 1.609957788897737 alphas[2] + 1.587318580622966 alphas[3] - alphas[4]
   + f_var[62] = 0
 R623: - 2156.17943612574 Fraction[62] + Q0_var[62] = 0
 R624: Q0_var[62] + Lost_0_expr_62 = 56.09568451910415
 R625: Q0_var[62] + Q_hat_adjusted_62 = 675.5019013807228
 R626: Sold_0[62] - Q0_var[62] - Q1_var[62] + total_stock_second_period_62
   = 0
 R627: total_stock_second_period_62 + Lost_1_expr_62 = 539.200158408942
 R628: Sold_1[62] + Left_1[62] - total_stock_second_period_62 = 0
 R629: - 700 Sold_0[62] - 700 Sold_1[62] + 700 Lost_0[62] + 700 Lost_1[62]
   + 300 Left_1[62] + profits_vars[62] = 0
 Fraction_lower_bound_63: Fraction[63] >= 0
 Fraction_upper_bound_63: Fraction[63] <= 1
 R632: 1.253324683762242 alphas[0] - 1.317272369956665 alphas[1]
   + 1.590142717409528 alphas[2] + 6.8268185141650314e-04 alphas[3]
   - alphas[4] + f_var[63] = 0
 R633: - 2156.17943612574 Fraction[63] + Q0_var[63] = 0
 R634: Q0_var[63] + Lost_0_expr_63 = 109.074867204533
 R635: Q0_var[63] + Q_hat_adjusted_63 = 686.9788995625934
 R636: Sold_0[63] - Q0_var[63] - Q1_var[63] + total_stock_second_period_63
   = 0
 R637: total_stock_second_period_63 + Lost_1_expr_63 = 353.153108878996
 R638: Sold_1[63] + Left_1[63] - total_stock_second_period_63 = 0
 R639: - 700 Sold_0[63] - 700 Sold_1[63] + 700 Lost_0[63] + 700 Lost_1[63]
   + 300 Left_1[63] + profits_vars[63] = 0
 Fraction_lower_bound_64: Fraction[64] >= 0
 Fraction_upper_bound_64: Fraction[64] <= 1
 R642: 1.046714789240151 alphas[0] - 0.059526169248796 alphas[1]
   - 0.048362002673092 alphas[2] - 0.1081836138137229 alphas[3] - alphas[4]
   + f_var[64] = 0
 R643: - 2156.17943612574 Fraction[64] + Q0_var[64] = 0
 R644: Q0_var[64] + Lost_0_expr_64 = 296.4050669163157
 R645: Q0_var[64] + Q_hat_adjusted_64 = 1841.736423916871
 R646: Sold_0[64] - Q0_var[64] - Q1_var[64] + total_stock_second_period_64
   = 0
 R647: total_stock_second_period_64 + Lost_1_expr_64 = 1202.570418835342
 R648: Sold_1[64] + Left_1[64] - total_stock_second_period_64 = 0
 R649: - 700 Sold_0[64] - 700 Sold_1[64] + 700 Lost_0[64] + 700 Lost_1[64]
   + 300 Left_1[64] + profits_vars[64] = 0
 Fraction_lower_bound_65: Fraction[65] >= 0
 Fraction_upper_bound_65: Fraction[65] <= 1
 R652: 0.4376948110515189 alphas[0] - 1.489584777425653 alphas[1]
   + 1.354345371039576 alphas[2] + 0.586830504051697 alphas[3] - alphas[4]
   + f_var[65] = 0
 R653: - 2156.17943612574 Fraction[65] + Q0_var[65] = 0
 R654: Q0_var[65] + Lost_0_expr_65 = 70.72097073381131
 R655: Q0_var[65] + Q_hat_adjusted_65 = 806.4988661625708
 R656: Sold_0[65] - Q0_var[65] - Q1_var[65] + total_stock_second_period_65
   = 0
 R657: total_stock_second_period_65 + Lost_1_expr_65 = 808.7000449059769
 R658: Sold_1[65] + Left_1[65] - total_stock_second_period_65 = 0
 R659: - 700 Sold_0[65] - 700 Sold_1[65] + 700 Lost_0[65] + 700 Lost_1[65]
   + 300 Left_1[65] + profits_vars[65] = 0
 Fraction_lower_bound_66: Fraction[66] >= 0
 Fraction_upper_bound_66: Fraction[66] <= 1
 R662: - 1.162373673784282 alphas[0] - 0.1912767080796875 alphas[1]
   + 1.459944028431884 alphas[2] + 1.499445459947536 alphas[3] - alphas[4]
   + f_var[66] = 0
 R663: - 2156.17943612574 Fraction[66] + Q0_var[66] = 0
 R664: Q0_var[66] + Lost_0_expr_66 = 124.2284859013619
 R665: Q0_var[66] + Q_hat_adjusted_66 = 786.8122333518088
 R666: Sold_0[66] - Q0_var[66] - Q1_var[66] + total_stock_second_period_66
   = 0
 R667: total_stock_second_period_66 + Lost_1_expr_66 = 567.8551966448016
 R668: Sold_1[66] + Left_1[66] - total_stock_second_period_66 = 0
 R669: - 700 Sold_0[66] - 700 Sold_1[66] + 700 Lost_0[66] + 700 Lost_1[66]
   + 300 Left_1[66] + profits_vars[66] = 0
 Fraction_lower_bound_67: Fraction[67] >= 0
 Fraction_upper_bound_67: Fraction[67] <= 1
 R672: 1.39866657343461 alphas[0] - 1.492849864142074 alphas[1]
   - 1.537487642686873 alphas[2] + 0.9605876550458219 alphas[3] - alphas[4]
   + f_var[67] = 0
 R673: - 2156.17943612574 Fraction[67] + Q0_var[67] = 0
 R674: Q0_var[67] + Lost_0_expr_67 = 665.7984071163157
 R675: Q0_var[67] + Q_hat_adjusted_67 = 3023.138504228752
 R676: Sold_0[67] - Q0_var[67] - Q1_var[67] + total_stock_second_period_67
   = 0
 R677: total_stock_second_period_67 + Lost_1_expr_67 = 2660.793811367604
 R678: Sold_1[67] + Left_1[67] - total_stock_second_period_67 = 0
 R679: - 700 Sold_0[67] - 700 Sold_1[67] + 700 Lost_0[67] + 700 Lost_1[67]
   + 300 Left_1[67] + profits_vars[67] = 0
 Fraction_lower_bound_68: Fraction[68] >= 0
 Fraction_upper_bound_68: Fraction[68] <= 1
 R682: - 1.222346613777015 alphas[0] - 1.174519908604492 alphas[1]
   - 1.188347151441529 alphas[2] + 0.5267191515358685 alphas[3] - alphas[4]
   + f_var[68] = 0
 R683: - 2156.17943612574 Fraction[68] + Q0_var[68] = 0
 R684: Q0_var[68] + Lost_0_expr_68 = 561.684926219236
 R685: Q0_var[68] + Q_hat_adjusted_68 = 2734.915222678407
 R686: Sold_0[68] - Q0_var[68] - Q1_var[68] + total_stock_second_period_68
   = 0
 R687: total_stock_second_period_68 + Lost_1_expr_68 = 1926.346080803945
 R688: Sold_1[68] + Left_1[68] - total_stock_second_period_68 = 0
 R689: - 700 Sold_0[68] - 700 Sold_1[68] + 700 Lost_0[68] + 700 Lost_1[68]
   + 300 Left_1[68] + profits_vars[68] = 0
 Fraction_lower_bound_69: Fraction[69] >= 0
 Fraction_upper_bound_69: Fraction[69] <= 1
 R692: 1.402214596088885 alphas[0] - 1.646139552440573 alphas[1]
   + 1.129035953192983 alphas[2] + 0.2673652817236263 alphas[3] - alphas[4]
   + f_var[69] = 0
 R693: - 2156.17943612574 Fraction[69] + Q0_var[69] = 0
 R694: Q0_var[69] + Lost_0_expr_69 = 218.0246496293819
 R695: Q0_var[69] + Q_hat_adjusted_69 = 1043.988556956084
 R696: Sold_0[69] - Q0_var[69] - Q1_var[69] + total_stock_second_period_69
   = 0
 R697: total_stock_second_period_69 + Lost_1_expr_69 = 833.1122346048919
 R698: Sold_1[69] + Left_1[69] - total_stock_second_period_69 = 0
 R699: - 700 Sold_0[69] - 700 Sold_1[69] + 700 Lost_0[69] + 700 Lost_1[69]
   + 300 Left_1[69] + profits_vars[69] = 0
 Fraction_lower_bound_70: Fraction[70] >= 0
 Fraction_upper_bound_70: Fraction[70] <= 1
 R702: - 1.712393995342048 alphas[0] - 1.480842863851215 alphas[1]
   + 1.625186275629722 alphas[2] - 0.2562790444665618 alphas[3] - alphas[4]
   + f_var[70] = 0
 R703: - 2156.17943612574 Fraction[70] + Q0_var[70] = 0
 R704: Q0_var[70] + Lost_0_expr_70 = 236.213596691663
 R705: Q0_var[70] + Q_hat_adjusted_70 = 717.5779878796009
 R706: Sold_0[70] - Q0_var[70] - Q1_var[70] + total_stock_second_period_70
   = 0
 R707: total_stock_second_period_70 + Lost_1_expr_70 = 583.9643147548329
 R708: Sold_1[70] + Left_1[70] - total_stock_second_period_70 = 0
 R709: - 700 Sold_0[70] - 700 Sold_1[70] + 700 Lost_0[70] + 700 Lost_1[70]
   + 300 Left_1[70] + profits_vars[70] = 0
 Fraction_lower_bound_71: Fraction[71] >= 0
 Fraction_upper_bound_71: Fraction[71] <= 1
 R712: 0.0841688660974765 alphas[0] + 1.530908503101651 alphas[1]
   - 1.459313909070566 alphas[2] - 0.6981759431533224 alphas[3] - alphas[4]
   + f_var[71] = 0
 R713: - 2156.17943612574 Fraction[71] + Q0_var[71] = 0
 R714: Q0_var[71] + Lost_0_expr_71 = 706.6521279301862
 R715: Q0_var[71] + Q_hat_adjusted_71 = 2999.773147026277
 R716: Sold_0[71] - Q0_var[71] - Q1_var[71] + total_stock_second_period_71
   = 0
 R717: total_stock_second_period_71 + Lost_1_expr_71 = 2189.636400534236
 R718: Sold_1[71] + Left_1[71] - total_stock_second_period_71 = 0
 R719: - 700 Sold_0[71] - 700 Sold_1[71] + 700 Lost_0[71] + 700 Lost_1[71]
   + 300 Left_1[71] + profits_vars[71] = 0
 Fraction_lower_bound_72: Fraction[72] >= 0
 Fraction_upper_bound_72: Fraction[72] <= 1
 R722: - 1.71346110085595 alphas[0] + 1.0580989987241 alphas[1]
   + 0.7150455781587881 alphas[2] + 0.6623558594754755 alphas[3]
   - alphas[4] + f_var[72] = 0
 R723: - 2156.17943612574 Fraction[72] + Q0_var[72] = 0
 R724: Q0_var[72] + Lost_0_expr_72 = 196.8179581575923
 R725: Q0_var[72] + Q_hat_adjusted_72 = 1293.742870365653
 R726: Sold_0[72] - Q0_var[72] - Q1_var[72] + total_stock_second_period_72
   = 0
 R727: total_stock_second_period_72 + Lost_1_expr_72 = 1006.990015484635
 R728: Sold_1[72] + Left_1[72] - total_stock_second_period_72 = 0
 R729: - 700 Sold_0[72] - 700 Sold_1[72] + 700 Lost_0[72] + 700 Lost_1[72]
   + 300 Left_1[72] + profits_vars[72] = 0
 Fraction_lower_bound_73: Fraction[73] >= 0
 Fraction_upper_bound_73: Fraction[73] <= 1
 R732: - 0.3976697952331758 alphas[0] + 0.3275668248718018 alphas[1]
   + 1.018012804597675 alphas[2] + 0.6242992324859348 alphas[3] - alphas[4]
   + f_var[73] = 0
 R733: - 2156.17943612574 Fraction[73] + Q0_var[73] = 0
 R734: Q0_var[73] + Lost_0_expr_73 = 276.246735988532
 R735: Q0_var[73] + Q_hat_adjusted_73 = 1153.09072541669
 R736: Sold_0[73] - Q0_var[73] - Q1_var[73] + total_stock_second_period_73
   = 0
 R737: total_stock_second_period_73 + Lost_1_expr_73 = 724.1258080325795
 R738: Sold_1[73] + Left_1[73] - total_stock_second_period_73 = 0
 R739: - 700 Sold_0[73] - 700 Sold_1[73] + 700 Lost_0[73] + 700 Lost_1[73]
   + 300 Left_1[73] + profits_vars[73] = 0
 Fraction_lower_bound_74: Fraction[74] >= 0
 Fraction_upper_bound_74: Fraction[74] <= 1
 R742: - 0.873224273766533 alphas[0] - 1.591122969612254 alphas[1]
   - 0.6393195737650093 alphas[2] + 1.663832077593131 alphas[3] - alphas[4]
   + f_var[74] = 0
 R743: - 2156.17943612574 Fraction[74] + Q0_var[74] = 0
 R744: Q0_var[74] + Lost_0_expr_74 = 380.0599411317962
 R745: Q0_var[74] + Q_hat_adjusted_74 = 2270.19255589433
 R746: Sold_0[74] - Q0_var[74] - Q1_var[74] + total_stock_second_period_74
   = 0
 R747: total_stock_second_period_74 + Lost_1_expr_74 = 1582.648076436338
 R748: Sold_1[74] + Left_1[74] - total_stock_second_period_74 = 0
 R749: - 700 Sold_0[74] - 700 Sold_1[74] + 700 Lost_0[74] + 700 Lost_1[74]
   + 300 Left_1[74] + profits_vars[74] = 0
 Fraction_lower_bound_75: Fraction[75] >= 0
 Fraction_upper_bound_75: Fraction[75] <= 1
 R752: 1.603557319033532 alphas[0] + 0.6312326726982082 alphas[1]
   + 1.270389728158603 alphas[2] + 1.592628237882033 alphas[3] - alphas[4]
   + f_var[75] = 0
 R753: - 2156.17943612574 Fraction[75] + Q0_var[75] = 0
 R754: Q0_var[75] + Lost_0_expr_75 = 77.12931336098755
 R755: Q0_var[75] + Q_hat_adjusted_75 = 863.8976357400202
 R756: Sold_0[75] - Q0_var[75] - Q1_var[75] + total_stock_second_period_75
   = 0
 R757: total_stock_second_period_75 + Lost_1_expr_75 = 817.9594699824568
 R758: Sold_1[75] + Left_1[75] - total_stock_second_period_75 = 0
 R759: - 700 Sold_0[75] - 700 Sold_1[75] + 700 Lost_0[75] + 700 Lost_1[75]
   + 300 Left_1[75] + profits_vars[75] = 0
 Fraction_lower_bound_76: Fraction[76] >= 0
 Fraction_upper_bound_76: Fraction[76] <= 1
 R762: 0.7416628101996238 alphas[0] + 1.106847203169944 alphas[1]
   + 0.7598729318423186 alphas[2] + 0.8633977655134649 alphas[3]
   - alphas[4] + f_var[76] = 0
 R763: - 2156.17943612574 Fraction[76] + Q0_var[76] = 0
 R764: Q0_var[76] + Lost_0_expr_76 = 250.1657757249098
 R765: Q0_var[76] + Q_hat_adjusted_76 = 1299.510607877612
 R766: Sold_0[76] - Q0_var[76] - Q1_var[76] + total_stock_second_period_76
   = 0
 R767: total_stock_second_period_76 + Lost_1_expr_76 = 743.2452575472593
 R768: Sold_1[76] + Left_1[76] - total_stock_second_period_76 = 0
 R769: - 700 Sold_0[76] - 700 Sold_1[76] + 700 Lost_0[76] + 700 Lost_1[76]
   + 300 Left_1[76] + profits_vars[76] = 0
 Fraction_lower_bound_77: Fraction[77] >= 0
 Fraction_upper_bound_77: Fraction[77] <= 1
 R772: 1.316958308616101 alphas[0] - 1.366751816736344 alphas[1]
   - 1.568883346371729 alphas[2] - 0.7581874245596035 alphas[3] - alphas[4]
   + f_var[77] = 0
 R773: - 2156.17943612574 Fraction[77] + Q0_var[77] = 0
 R774: Q0_var[77] + Lost_0_expr_77 = 669.803650277536
 R775: Q0_var[77] + Q_hat_adjusted_77 = 3045.619326183199
 R776: Sold_0[77] - Q0_var[77] - Q1_var[77] + total_stock_second_period_77
   = 0
 R777: total_stock_second_period_77 + Lost_1_expr_77 = 2445.191357157138
 R778: Sold_1[77] + Left_1[77] - total_stock_second_period_77 = 0
 R779: - 700 Sold_0[77] - 700 Sold_1[77] + 700 Lost_0[77] + 700 Lost_1[77]
   + 300 Left_1[77] + profits_vars[77] = 0
 Fraction_lower_bound_78: Fraction[78] >= 0
 Fraction_upper_bound_78: Fraction[78] <= 1
 R782: 0.6944914731922428 alphas[0] + 0.1324298610493309 alphas[1]
   - 0.4687210294958513 alphas[2] - 0.3819495398297618 alphas[3]
   - alphas[4] + f_var[78] = 0
 R783: - 2156.17943612574 Fraction[78] + Q0_var[78] = 0
 R784: Q0_var[78] + Lost_0_expr_78 = 280.7808026233366
 R785: Q0_var[78] + Q_hat_adjusted_78 = 2098.301394786328
 R786: Sold_0[78] - Q0_var[78] - Q1_var[78] + total_stock_second_period_78
   = 0
 R787: total_stock_second_period_78 + Lost_1_expr_78 = 1633.052065568451
 R788: Sold_1[78] + Left_1[78] - total_stock_second_period_78 = 0
 R789: - 700 Sold_0[78] - 700 Sold_1[78] + 700 Lost_0[78] + 700 Lost_1[78]
   + 300 Left_1[78] + profits_vars[78] = 0
 Fraction_lower_bound_79: Fraction[79] >= 0
 Fraction_upper_bound_79: Fraction[79] <= 1
 R792: 1.322154890952527 alphas[0] - 1.440263948784893 alphas[1]
   - 0.0793576827348264 alphas[2] - 0.8574947130391086 alphas[3]
   - alphas[4] + f_var[79] = 0
 R793: - 2156.17943612574 Fraction[79] + Q0_var[79] = 0
 R794: Q0_var[79] + Lost_0_expr_79 = 465.9637668013111
 R795: Q0_var[79] + Q_hat_adjusted_79 = 1970.184572133232
 R796: Sold_0[79] - Q0_var[79] - Q1_var[79] + total_stock_second_period_79
   = 0
 R797: total_stock_second_period_79 + Lost_1_expr_79 = 1564.405145323783
 R798: Sold_1[79] + Left_1[79] - total_stock_second_period_79 = 0
 R799: - 700 Sold_0[79] - 700 Sold_1[79] + 700 Lost_0[79] + 700 Lost_1[79]
   + 300 Left_1[79] + profits_vars[79] = 0
 Fraction_lower_bound_80: Fraction[80] >= 0
 Fraction_upper_bound_80: Fraction[80] <= 1
 R802: 0.6172137035200134 alphas[0] + 1.107655486821045 alphas[1]
   + 0.3230326082044016 alphas[2] - 0.8545717504108536 alphas[3]
   - alphas[4] + f_var[80] = 0
 R803: - 2156.17943612574 Fraction[80] + Q0_var[80] = 0
 R804: Q0_var[80] + Lost_0_expr_80 = 238.4104139152425
 R805: Q0_var[80] + Q_hat_adjusted_80 = 1568.987931639154
 R806: Sold_0[80] - Q0_var[80] - Q1_var[80] + total_stock_second_period_80
   = 0
 R807: total_stock_second_period_80 + Lost_1_expr_80 = 1239.036119925152
 R808: Sold_1[80] + Left_1[80] - total_stock_second_period_80 = 0
 R809: - 700 Sold_0[80] - 700 Sold_1[80] + 700 Lost_0[80] + 700 Lost_1[80]
   + 300 Left_1[80] + profits_vars[80] = 0
 Fraction_lower_bound_81: Fraction[81] >= 0
 Fraction_upper_bound_81: Fraction[81] <= 1
 R812: 0.2765876195613583 alphas[0] - 0.5993306080836938 alphas[1]
   - 0.9113604653983925 alphas[2] - 1.149185938311494 alphas[3] - alphas[4]
   + f_var[81] = 0
 R813: - 2156.17943612574 Fraction[81] + Q0_var[81] = 0
 R814: Q0_var[81] + Lost_0_expr_81 = 487.4757754160669
 R815: Q0_var[81] + Q_hat_adjusted_81 = 2511.638373844792
 R816: Sold_0[81] - Q0_var[81] - Q1_var[81] + total_stock_second_period_81
   = 0
 R817: total_stock_second_period_81 + Lost_1_expr_81 = 1684.120878757362
 R818: Sold_1[81] + Left_1[81] - total_stock_second_period_81 = 0
 R819: - 700 Sold_0[81] - 700 Sold_1[81] + 700 Lost_0[81] + 700 Lost_1[81]
   + 300 Left_1[81] + profits_vars[81] = 0
 Fraction_lower_bound_82: Fraction[82] >= 0
 Fraction_upper_bound_82: Fraction[82] <= 1
 R822: 1.515252971421249 alphas[0] + 0.1533227070571766 alphas[1]
   + 1.165321307337225 alphas[2] - 0.6912104125013629 alphas[3] - alphas[4]
   + f_var[82] = 0
 R823: - 2156.17943612574 Fraction[82] + Q0_var[82] = 0
 R824: Q0_var[82] + Lost_0_expr_82 = 191.4951348800481
 R825: Q0_var[82] + Q_hat_adjusted_82 = 1004.747181350988
 R826: Sold_0[82] - Q0_var[82] - Q1_var[82] + total_stock_second_period_82
   = 0
 R827: total_stock_second_period_82 + Lost_1_expr_82 = 805.6324794119407
 R828: Sold_1[82] + Left_1[82] - total_stock_second_period_82 = 0
 R829: - 700 Sold_0[82] - 700 Sold_1[82] + 700 Lost_0[82] + 700 Lost_1[82]
   + 300 Left_1[82] + profits_vars[82] = 0
 Fraction_lower_bound_83: Fraction[83] >= 0
 Fraction_upper_bound_83: Fraction[83] <= 1
 R832: - 0.7076818724629808 alphas[0] + 1.393047828633475 alphas[1]
   - 1.543457202674104 alphas[2] - 1.56075910634691 alphas[3] - alphas[4]
   + f_var[83] = 0
 R833: - 2156.17943612574 Fraction[83] + Q0_var[83] = 0
 R834: Q0_var[83] + Lost_0_expr_83 = 586.8433741013578
 R835: Q0_var[83] + Q_hat_adjusted_83 = 2976.266223637631
 R836: Sold_0[83] - Q0_var[83] - Q1_var[83] + total_stock_second_period_83
   = 0
 R837: total_stock_second_period_83 + Lost_1_expr_83 = 2272.263269995019
 R838: Sold_1[83] + Left_1[83] - total_stock_second_period_83 = 0
 R839: - 700 Sold_0[83] - 700 Sold_1[83] + 700 Lost_0[83] + 700 Lost_1[83]
   + 300 Left_1[83] + profits_vars[83] = 0
 Fraction_lower_bound_84: Fraction[84] >= 0
 Fraction_upper_bound_84: Fraction[84] <= 1
 R842: - 0.2623670190209739 alphas[0] - 0.7197859180637243 alphas[1]
   - 1.195297472254883 alphas[2] - 1.291228297656285 alphas[3] - alphas[4]
   + f_var[84] = 0
 R843: - 2156.17943612574 Fraction[84] + Q0_var[84] = 0
 R844: Q0_var[84] + Lost_0_expr_84 = 496.4411566341943
 R845: Q0_var[84] + Q_hat_adjusted_84 = 2697.462177195157
 R846: Sold_0[84] - Q0_var[84] - Q1_var[84] + total_stock_second_period_84
   = 0
 R847: total_stock_second_period_84 + Lost_1_expr_84 = 2075.844551681394
 R848: Sold_1[84] + Left_1[84] - total_stock_second_period_84 = 0
 R849: - 700 Sold_0[84] - 700 Sold_1[84] + 700 Lost_0[84] + 700 Lost_1[84]
   + 300 Left_1[84] + profits_vars[84] = 0
 Fraction_lower_bound_85: Fraction[85] >= 0
 Fraction_upper_bound_85: Fraction[85] <= 1
 R852: 0.8032836717345447 alphas[0] + 0.8105923883780608 alphas[1]
   - 0.3972441819937966 alphas[2] - 0.2421293154075811 alphas[3]
   - alphas[4] + f_var[85] = 0
 R853: - 2156.17943612574 Fraction[85] + Q0_var[85] = 0
 R854: Q0_var[85] + Lost_0_expr_85 = 532.8488618728193
 R855: Q0_var[85] + Q_hat_adjusted_85 = 2214.700004014168
 R856: Sold_0[85] - Q0_var[85] - Q1_var[85] + total_stock_second_period_85
   = 0
 R857: total_stock_second_period_85 + Lost_1_expr_85 = 1610.547469894234
 R858: Sold_1[85] + Left_1[85] - total_stock_second_period_85 = 0
 R859: - 700 Sold_0[85] - 700 Sold_1[85] + 700 Lost_0[85] + 700 Lost_1[85]
   + 300 Left_1[85] + profits_vars[85] = 0
 Fraction_lower_bound_86: Fraction[86] >= 0
 Fraction_upper_bound_86: Fraction[86] <= 1
 R862: - 0.1089880464730707 alphas[0] + 1.51870231620292 alphas[1]
   + 1.486722009319007 alphas[2] - 0.246971329404053 alphas[3] - alphas[4]
   + f_var[86] = 0
 R863: - 2156.17943612574 Fraction[86] + Q0_var[86] = 0
 R864: Q0_var[86] + Lost_0_expr_86 = 60.70907708378834
 R865: Q0_var[86] + Q_hat_adjusted_86 = 716.8235148143833
 R866: Sold_0[86] - Q0_var[86] - Q1_var[86] + total_stock_second_period_86
   = 0
 R867: total_stock_second_period_86 + Lost_1_expr_86 = 608.1999388661843
 R868: Sold_1[86] + Left_1[86] - total_stock_second_period_86 = 0
 R869: - 700 Sold_0[86] - 700 Sold_1[86] + 700 Lost_0[86] + 700 Lost_1[86]
   + 300 Left_1[86] + profits_vars[86] = 0
 Fraction_lower_bound_87: Fraction[87] >= 0
 Fraction_upper_bound_87: Fraction[87] <= 1
 R872: 1.40984896939898 alphas[0] + 1.449423348562219 alphas[1]
   - 0.6699597270572338 alphas[2] + 1.614839317979502 alphas[3] - alphas[4]
   + f_var[87] = 0
 R873: - 2156.17943612574 Fraction[87] + Q0_var[87] = 0
 R874: Q0_var[87] + Lost_0_expr_87 = 369.1660006900256
 R875: Q0_var[87] + Q_hat_adjusted_87 = 2282.63469814785
 R876: Sold_0[87] - Q0_var[87] - Q1_var[87] + total_stock_second_period_87
   = 0
 R877: total_stock_second_period_87 + Lost_1_expr_87 = 1926.039820860612
 R878: Sold_1[87] + Left_1[87] - total_stock_second_period_87 = 0
 R879: - 700 Sold_0[87] - 700 Sold_1[87] + 700 Lost_0[87] + 700 Lost_1[87]
   + 300 Left_1[87] + profits_vars[87] = 0
 Fraction_lower_bound_88: Fraction[88] >= 0
 Fraction_upper_bound_88: Fraction[88] <= 1
 R882: - 0.2954286200985933 alphas[0] + 1.4455579463565 alphas[1]
   - 1.366276249257802 alphas[2] - 1.626357726715828 alphas[3] - alphas[4]
   + f_var[88] = 0
 R883: - 2156.17943612574 Fraction[88] + Q0_var[88] = 0
 R884: Q0_var[88] + Lost_0_expr_88 = 407.9740012446229
 R885: Q0_var[88] + Q_hat_adjusted_88 = 2749.13493026452
 R886: Sold_0[88] - Q0_var[88] - Q1_var[88] + total_stock_second_period_88
   = 0
 R887: total_stock_second_period_88 + Lost_1_expr_88 = 2441.950956662425
 R888: Sold_1[88] + Left_1[88] - total_stock_second_period_88 = 0
 R889: - 700 Sold_0[88] - 700 Sold_1[88] + 700 Lost_0[88] + 700 Lost_1[88]
   + 300 Left_1[88] + profits_vars[88] = 0
 Fraction_lower_bound_89: Fraction[89] >= 0
 Fraction_upper_bound_89: Fraction[89] <= 1
 R892: - 1.545536185387975 alphas[0] - 1.439777852858012 alphas[1]
   - 0.0921390555442572 alphas[2] + 1.287751545780062 alphas[3] - alphas[4]
   + f_var[89] = 0
 R893: - 2156.17943612574 Fraction[89] + Q0_var[89] = 0
 R894: Q0_var[89] + Lost_0_expr_89 = 333.2553660523799
 R895: Q0_var[89] + Q_hat_adjusted_89 = 1893.140250062993
 R896: Sold_0[89] - Q0_var[89] - Q1_var[89] + total_stock_second_period_89
   = 0
 R897: total_stock_second_period_89 + Lost_1_expr_89 = 1296.321948003394
 R898: Sold_1[89] + Left_1[89] - total_stock_second_period_89 = 0
 R899: - 700 Sold_0[89] - 700 Sold_1[89] + 700 Lost_0[89] + 700 Lost_1[89]
   + 300 Left_1[89] + profits_vars[89] = 0
 Fraction_lower_bound_90: Fraction[90] >= 0
 Fraction_upper_bound_90: Fraction[90] <= 1
 R902: 0.6151413108553347 alphas[0] - 0.8677607094953069 alphas[1]
   + 0.4662557753902354 alphas[2] - 0.7585309492470805 alphas[3]
   - alphas[4] + f_var[90] = 0
 R903: - 2156.17943612574 Fraction[90] + Q0_var[90] = 0
 R904: Q0_var[90] + Lost_0_expr_90 = 325.5925592186188
 R905: Q0_var[90] + Q_hat_adjusted_90 = 1534.092758048565
 R906: Sold_0[90] - Q0_var[90] - Q1_var[90] + total_stock_second_period_90
   = 0
 R907: total_stock_second_period_90 + Lost_1_expr_90 = 1035.901823795766
 R908: Sold_1[90] + Left_1[90] - total_stock_second_period_90 = 0
 R909: - 700 Sold_0[90] - 700 Sold_1[90] + 700 Lost_0[90] + 700 Lost_1[90]
   + 300 Left_1[90] + profits_vars[90] = 0
 Fraction_lower_bound_91: Fraction[91] >= 0
 Fraction_upper_bound_91: Fraction[91] <= 1
 R912: - 0.6190165365307102 alphas[0] - 1.40682116062353 alphas[1]
   + 1.233108371212938 alphas[2] - 1.029682924609413 alphas[3] - alphas[4]
   + f_var[91] = 0
 R913: - 2156.17943612574 Fraction[91] + Q0_var[91] = 0
 R914: Q0_var[91] + Lost_0_expr_91 = 80.3238758800838
 R915: Q0_var[91] + Q_hat_adjusted_91 = 892.5109556890441
 R916: Sold_0[91] - Q0_var[91] - Q1_var[91] + total_stock_second_period_91
   = 0
 R917: total_stock_second_period_91 + Lost_1_expr_91 = 684.3430090950701
 R918: Sold_1[91] + Left_1[91] - total_stock_second_period_91 = 0
 R919: - 700 Sold_0[91] - 700 Sold_1[91] + 700 Lost_0[91] + 700 Lost_1[91]
   + 300 Left_1[91] + profits_vars[91] = 0
 Fraction_lower_bound_92: Fraction[92] >= 0
 Fraction_upper_bound_92: Fraction[92] <= 1
 R922: 1.275914339133635 alphas[0] - 0.6384017998977272 alphas[1]
   - 0.8471144362241935 alphas[2] + 0.4760618774317884 alphas[3]
   - alphas[4] + f_var[92] = 0
 R923: - 2156.17943612574 Fraction[92] + Q0_var[92] = 0
 R924: Q0_var[92] + Lost_0_expr_92 = 538.589861310528
 R925: Q0_var[92] + Q_hat_adjusted_92 = 2503.689019997249
 R926: Sold_0[92] - Q0_var[92] - Q1_var[92] + total_stock_second_period_92
   = 0
 R927: total_stock_second_period_92 + Lost_1_expr_92 = 1525.291727435214
 R928: Sold_1[92] + Left_1[92] - total_stock_second_period_92 = 0
 R929: - 700 Sold_0[92] - 700 Sold_1[92] + 700 Lost_0[92] + 700 Lost_1[92]
   + 300 Left_1[92] + profits_vars[92] = 0
 Fraction_lower_bound_93: Fraction[93] >= 0
 Fraction_upper_bound_93: Fraction[93] <= 1
 R932: - 0.7920782133804717 alphas[0] - 0.1488791589709158 alphas[1]
   - 1.273635098078513 alphas[2] + 1.068337004558573 alphas[3] - alphas[4]
   + f_var[93] = 0
 R933: - 2156.17943612574 Fraction[93] + Q0_var[93] = 0
 R934: Q0_var[93] + Lost_0_expr_93 = 371.0826538671789
 R935: Q0_var[93] + Q_hat_adjusted_93 = 2666.712563965829
 R936: Sold_0[93] - Q0_var[93] - Q1_var[93] + total_stock_second_period_93
   = 0
 R937: total_stock_second_period_93 + Lost_1_expr_93 = 2271.819961610808
 R938: Sold_1[93] + Left_1[93] - total_stock_second_period_93 = 0
 R939: - 700 Sold_0[93] - 700 Sold_1[93] + 700 Lost_0[93] + 700 Lost_1[93]
   + 300 Left_1[93] + profits_vars[93] = 0
 Fraction_lower_bound_94: Fraction[94] >= 0
 Fraction_upper_bound_94: Fraction[94] <= 1
 R942: 0.7183159027204433 alphas[0] + 0.6161783464923033 alphas[1]
   + 1.298749959841151 alphas[2] + 1.398057408979115 alphas[3] - alphas[4]
   + f_var[94] = 0
 R943: - 2156.17943612574 Fraction[94] + Q0_var[94] = 0
 R944: Q0_var[94] + Lost_0_expr_94 = 181.0184262136766
 R945: Q0_var[94] + Q_hat_adjusted_94 = 911.8633560694469
 R946: Sold_0[94] - Q0_var[94] - Q1_var[94] + total_stock_second_period_94
   = 0
 R947: total_stock_second_period_94 + Lost_1_expr_94 = 641.591686240127
 R948: Sold_1[94] + Left_1[94] - total_stock_second_period_94 = 0
 R949: - 700 Sold_0[94] - 700 Sold_1[94] + 700 Lost_0[94] + 700 Lost_1[94]
   + 300 Left_1[94] + profits_vars[94] = 0
 Fraction_lower_bound_95: Fraction[95] >= 0
 Fraction_upper_bound_95: Fraction[95] <= 1
 R952: 1.094090490711597 alphas[0] - 1.747272629222934 alphas[1]
   - 0.8604995119943858 alphas[2] - 0.7862915540720115 alphas[3]
   - alphas[4] + f_var[95] = 0
 R953: - 2156.17943612574 Fraction[95] + Q0_var[95] = 0
 R954: Q0_var[95] + Lost_0_expr_95 = 482.5269160686618
 R955: Q0_var[95] + Q_hat_adjusted_95 = 2476.207376719895
 R956: Sold_0[95] - Q0_var[95] - Q1_var[95] + total_stock_second_period_95
   = 0
 R957: total_stock_second_period_95 + Lost_1_expr_95 = 1603.177986769715
 R958: Sold_1[95] + Left_1[95] - total_stock_second_period_95 = 0
 R959: - 700 Sold_0[95] - 700 Sold_1[95] + 700 Lost_0[95] + 700 Lost_1[95]
   + 300 Left_1[95] + profits_vars[95] = 0
 Fraction_lower_bound_96: Fraction[96] >= 0
 Fraction_upper_bound_96: Fraction[96] <= 1
 R962: - 0.3328113759746041 alphas[0] + 0.4193955035383445 alphas[1]
   + 1.571441218862665 alphas[2] + 0.2085031619166977 alphas[3] - alphas[4]
   + f_var[96] = 0
 R963: - 2156.17943612574 Fraction[96] + Q0_var[96] = 0
 R964: Q0_var[96] + Lost_0_expr_96 = 147.7291206207728
 R965: Q0_var[96] + Q_hat_adjusted_96 = 718.4318387773219
 R966: Sold_0[96] - Q0_var[96] - Q1_var[96] + total_stock_second_period_96
   = 0
 R967: total_stock_second_period_96 + Lost_1_expr_96 = 402.2032392178372
 R968: Sold_1[96] + Left_1[96] - total_stock_second_period_96 = 0
 R969: - 700 Sold_0[96] - 700 Sold_1[96] + 700 Lost_0[96] + 700 Lost_1[96]
   + 300 Left_1[96] + profits_vars[96] = 0
 Fraction_lower_bound_97: Fraction[97] >= 0
 Fraction_upper_bound_97: Fraction[97] <= 1
 R972: 1.67106087095211 alphas[0] + 0.0497721995776152 alphas[1]
   - 1.54252994198387 alphas[2] - 0.669020551676746 alphas[3] - alphas[4]
   + f_var[97] = 0
 R973: - 2156.17943612574 Fraction[97] + Q0_var[97] = 0
 R974: Q0_var[97] + Lost_0_expr_97 = 719.3519782140543
 R975: Q0_var[97] + Q_hat_adjusted_97 = 3060.696690828257
 R976: Sold_0[97] - Q0_var[97] - Q1_var[97] + total_stock_second_period_97
   = 0
 R977: total_stock_second_period_97 + Lost_1_expr_97 = 2106.99266033027
 R978: Sold_1[97] + Left_1[97] - total_stock_second_period_97 = 0
 R979: - 700 Sold_0[97] - 700 Sold_1[97] + 700 Lost_0[97] + 700 Lost_1[97]
   + 300 Left_1[97] + profits_vars[97] = 0
 Fraction_lower_bound_98: Fraction[98] >= 0
 Fraction_upper_bound_98: Fraction[98] <= 1
 R982: - 1.190488467584209 alphas[0] + 0.3646774448736319 alphas[1]
   - 0.3887250873128066 alphas[2] - 1.34696735915278 alphas[3] - alphas[4]
   + f_var[98] = 0
 R983: - 2156.17943612574 Fraction[98] + Q0_var[98] = 0
 R984: Q0_var[98] + Lost_0_expr_98 = 393.7542105669801
 R985: Q0_var[98] + Q_hat_adjusted_98 = 2120.052938020309
 R986: Sold_0[98] - Q0_var[98] - Q1_var[98] + total_stock_second_period_98
   = 0
 R987: total_stock_second_period_98 + Lost_1_expr_98 = 1785.49396348099
 R988: Sold_1[98] + Left_1[98] - total_stock_second_period_98 = 0
 R989: - 700 Sold_0[98] - 700 Sold_1[98] + 700 Lost_0[98] + 700 Lost_1[98]
   + 300 Left_1[98] + profits_vars[98] = 0
 Fraction_lower_bound_99: Fraction[99] >= 0
 Fraction_upper_bound_99: Fraction[99] <= 1
 R992: 1.725586869381733 alphas[0] - 1.684446412015219 alphas[1]
   - 0.1834674203739427 alphas[2] + 1.237305543731072 alphas[3] - alphas[4]
   + f_var[99] = 0
 R993: - 2156.17943612574 Fraction[99] + Q0_var[99] = 0
 R994: Q0_var[99] + Lost_0_expr_99 = 405.1130514348309
 R995: Q0_var[99] + Q_hat_adjusted_99 = 1997.16693621135
 R996: Sold_0[99] - Q0_var[99] - Q1_var[99] + total_stock_second_period_99
   = 0
 R997: total_stock_second_period_99 + Lost_1_expr_99 = 1622.786544699135
 R998: Sold_1[99] + Left_1[99] - total_stock_second_period_99 = 0
 R999: - 700 Sold_0[99] - 700 Sold_1[99] + 700 Lost_0[99] + 700 Lost_1[99]
   + 300 Left_1[99] + profits_vars[99] = 0
 Fraction_lower_bound_100: Fraction[100] >= 0
 Fraction_upper_bound_100: Fraction[100] <= 1
 R1002: - 0.6558322330137908 alphas[0] + 1.05848620309569 alphas[1]
   + 0.6572744031789803 alphas[2] - 1.359617236590463 alphas[3] - alphas[4]
   + f_var[100] = 0
 R1003: - 2156.17943612574 Fraction[100] + Q0_var[100] = 0
 R1004: Q0_var[100] + Lost_0_expr_100 = 264.0053245797885
 R1005: Q0_var[100] + Q_hat_adjusted_100 = 1373.428152380167
 R1006: Sold_0[100] - Q0_var[100] - Q1_var[100]
   + total_stock_second_period_100 = 0
 R1007: total_stock_second_period_100 + Lost_1_expr_100
   = 1011.934648091743
 R1008: Sold_1[100] + Left_1[100] - total_stock_second_period_100 = 0
 R1009: - 700 Sold_0[100] - 700 Sold_1[100] + 700 Lost_0[100]
   + 700 Lost_1[100] + 300 Left_1[100] + profits_vars[100] = 0
 Fraction_lower_bound_101: Fraction[101] >= 0
 Fraction_upper_bound_101: Fraction[101] <= 1
 R1012: 0.78694405245293 alphas[0] + 0.5359501859789108 alphas[1]
   - 1.418746415864626 alphas[2] + 0.346408789917921 alphas[3] - alphas[4]
   + f_var[101] = 0
 R1013: - 2156.17943612574 Fraction[101] + Q0_var[101] = 0
 R1014: Q0_var[101] + Lost_0_expr_101 = 684.0315242476323
 R1015: Q0_var[101] + Q_hat_adjusted_101 = 2959.531884671857
 R1016: Sold_0[101] - Q0_var[101] - Q1_var[101]
   + total_stock_second_period_101 = 0
 R1017: total_stock_second_period_101 + Lost_1_expr_101
   = 2084.496258901394
 R1018: Sold_1[101] + Left_1[101] - total_stock_second_period_101 = 0
 R1019: - 700 Sold_0[101] - 700 Sold_1[101] + 700 Lost_0[101]
   + 700 Lost_1[101] + 300 Left_1[101] + profits_vars[101] = 0
 Fraction_lower_bound_102: Fraction[102] >= 0
 Fraction_upper_bound_102: Fraction[102] <= 1
 R1022: - 0.8588266831964657 alphas[0] - 0.6648689637820892 alphas[1]
   + 0.755755813896965 alphas[2] + 0.7039182781831694 alphas[3] - alphas[4]
   + f_var[102] = 0
 R1023: - 2156.17943612574 Fraction[102] + Q0_var[102] = 0
 R1024: Q0_var[102] + Lost_0_expr_102 = 224.6611250678646
 R1025: Q0_var[102] + Q_hat_adjusted_102 = 1285.705881821333
 R1026: Sold_0[102] - Q0_var[102] - Q1_var[102]
   + total_stock_second_period_102 = 0
 R1027: total_stock_second_period_102 + Lost_1_expr_102
   = 967.1215075403429
 R1028: Sold_1[102] + Left_1[102] - total_stock_second_period_102 = 0
 R1029: - 700 Sold_0[102] - 700 Sold_1[102] + 700 Lost_0[102]
   + 700 Lost_1[102] + 300 Left_1[102] + profits_vars[102] = 0
 Fraction_lower_bound_103: Fraction[103] >= 0
 Fraction_upper_bound_103: Fraction[103] <= 1
 R1032: - 1.661905244895716 alphas[0] + 1.438302194464185 alphas[1]
   - 1.087204688645368 alphas[2] - 1.439923707943722 alphas[3] - alphas[4]
   + f_var[103] = 0
 R1033: - 2156.17943612574 Fraction[103] + Q0_var[103] = 0
 R1034: Q0_var[103] + Lost_0_expr_103 = 590.0875281880088
 R1035: Q0_var[103] + Q_hat_adjusted_103 = 2688.994452513794
 R1036: Sold_0[103] - Q0_var[103] - Q1_var[103]
   + total_stock_second_period_103 = 0
 R1037: total_stock_second_period_103 + Lost_1_expr_103
   = 1957.951148038918
 R1038: Sold_1[103] + Left_1[103] - total_stock_second_period_103 = 0
 R1039: - 700 Sold_0[103] - 700 Sold_1[103] + 700 Lost_0[103]
   + 700 Lost_1[103] + 300 Left_1[103] + profits_vars[103] = 0
 Fraction_lower_bound_104: Fraction[104] >= 0
 Fraction_upper_bound_104: Fraction[104] <= 1
 R1042: 0.8621410123492088 alphas[0] - 0.4157519698489255 alphas[1]
   - 1.601708069797295 alphas[2] - 1.59988761848153 alphas[3] - alphas[4]
   + f_var[104] = 0
 R1043: - 2156.17943612574 Fraction[104] + Q0_var[104] = 0
 R1044: Q0_var[104] + Lost_0_expr_104 = 501.3013921662313
 R1045: Q0_var[104] + Q_hat_adjusted_104 = 2958.32426062537
 R1046: Sold_0[104] - Q0_var[104] - Q1_var[104]
   + total_stock_second_period_104 = 0
 R1047: total_stock_second_period_104 + Lost_1_expr_104
   = 2368.083219029008
 R1048: Sold_1[104] + Left_1[104] - total_stock_second_period_104 = 0
 R1049: - 700 Sold_0[104] - 700 Sold_1[104] + 700 Lost_0[104]
   + 700 Lost_1[104] + 300 Left_1[104] + profits_vars[104] = 0
 Fraction_lower_bound_105: Fraction[105] >= 0
 Fraction_upper_bound_105: Fraction[105] <= 1
 R1052: - 0.2961745419276156 alphas[0] + 0.0259250529704071 alphas[1]
   - 0.9405195037587528 alphas[2] + 1.061479337757096 alphas[3] - alphas[4]
   + f_var[105] = 0
 R1053: - 2156.17943612574 Fraction[105] + Q0_var[105] = 0
 R1054: Q0_var[105] + Lost_0_expr_105 = 622.251132397608
 R1055: Q0_var[105] + Q_hat_adjusted_105 = 2616.603807918163
 R1056: Sold_0[105] - Q0_var[105] - Q1_var[105]
   + total_stock_second_period_105 = 0
 R1057: total_stock_second_period_105 + Lost_1_expr_105
   = 1876.919481363728
 R1058: Sold_1[105] + Left_1[105] - total_stock_second_period_105 = 0
 R1059: - 700 Sold_0[105] - 700 Sold_1[105] + 700 Lost_0[105]
   + 700 Lost_1[105] + 300 Left_1[105] + profits_vars[105] = 0
 Fraction_lower_bound_106: Fraction[106] >= 0
 Fraction_upper_bound_106: Fraction[106] <= 1
 R1062: - 0.3523722819716529 alphas[0] + 0.6858581268627546 alphas[1]
   - 0.0738250377629075 alphas[2] - 0.4778299280991928 alphas[3]
   - alphas[4] + f_var[106] = 0
 R1063: - 2156.17943612574 Fraction[106] + Q0_var[106] = 0
 R1064: Q0_var[106] + Lost_0_expr_106 = 297.5778032439039
 R1065: Q0_var[106] + Q_hat_adjusted_106 = 1858.637266105869
 R1066: Sold_0[106] - Q0_var[106] - Q1_var[106]
   + total_stock_second_period_106 = 0
 R1067: total_stock_second_period_106 + Lost_1_expr_106
   = 1523.209215594627
 R1068: Sold_1[106] + Left_1[106] - total_stock_second_period_106 = 0
 R1069: - 700 Sold_0[106] - 700 Sold_1[106] + 700 Lost_0[106]
   + 700 Lost_1[106] + 300 Left_1[106] + profits_vars[106] = 0
 Fraction_lower_bound_107: Fraction[107] >= 0
 Fraction_upper_bound_107: Fraction[107] <= 1
 R1072: - 0.2823576182275128 alphas[0] + 0.8412142229242753 alphas[1]
   + 1.410779761494625 alphas[2] + 1.372056065878673 alphas[3] - alphas[4]
   + f_var[107] = 0
 R1073: - 2156.17943612574 Fraction[107] + Q0_var[107] = 0
 R1074: Q0_var[107] + Lost_0_expr_107 = 147.0119923328621
 R1075: Q0_var[107] + Q_hat_adjusted_107 = 822.029435813976
 R1076: Sold_0[107] - Q0_var[107] - Q1_var[107]
   + total_stock_second_period_107 = 0
 R1077: total_stock_second_period_107 + Lost_1_expr_107
   = 642.4255277919862
 R1078: Sold_1[107] + Left_1[107] - total_stock_second_period_107 = 0
 R1079: - 700 Sold_0[107] - 700 Sold_1[107] + 700 Lost_0[107]
   + 700 Lost_1[107] + 300 Left_1[107] + profits_vars[107] = 0
 Fraction_lower_bound_108: Fraction[108] >= 0
 Fraction_upper_bound_108: Fraction[108] <= 1
 R1082: 0.952963644374102 alphas[0] - 0.0997123744235002 alphas[1]
   + 0.0788460985413931 alphas[2] - 0.4579162403567505 alphas[3]
   - alphas[4] + f_var[108] = 0
 R1083: - 2156.17943612574 Fraction[108] + Q0_var[108] = 0
 R1084: Q0_var[108] + Lost_0_expr_108 = 295.8235141190507
 R1085: Q0_var[108] + Q_hat_adjusted_108 = 1760.689330561308
 R1086: Sold_0[108] - Q0_var[108] - Q1_var[108]
   + total_stock_second_period_108 = 0
 R1087: total_stock_second_period_108 + Lost_1_expr_108 = 1256.63588680922
 R1088: Sold_1[108] + Left_1[108] - total_stock_second_period_108 = 0
 R1089: - 700 Sold_0[108] - 700 Sold_1[108] + 700 Lost_0[108]
   + 700 Lost_1[108] + 300 Left_1[108] + profits_vars[108] = 0
 Fraction_lower_bound_109: Fraction[109] >= 0
 Fraction_upper_bound_109: Fraction[109] <= 1
 R1092: - 1.628509341915556 alphas[0] + 0.3998390453176563 alphas[1]
   + 0.1908115290355662 alphas[2] - 1.050852707252724 alphas[3] - alphas[4]
   + f_var[109] = 0
 R1093: - 2156.17943612574 Fraction[109] + Q0_var[109] = 0
 R1094: Q0_var[109] + Lost_0_expr_109 = 506.005740669968
 R1095: Q0_var[109] + Q_hat_adjusted_109 = 1824.536309502311
 R1096: Sold_0[109] - Q0_var[109] - Q1_var[109]
   + total_stock_second_period_109 = 0
 R1097: total_stock_second_period_109 + Lost_1_expr_109
   = 1317.311720029328
 R1098: Sold_1[109] + Left_1[109] - total_stock_second_period_109 = 0
 R1099: - 700 Sold_0[109] - 700 Sold_1[109] + 700 Lost_0[109]
   + 700 Lost_1[109] + 300 Left_1[109] + profits_vars[109] = 0
 Fraction_lower_bound_110: Fraction[110] >= 0
 Fraction_upper_bound_110: Fraction[110] <= 1
 R1102: 0.1603245669941782 alphas[0] + 0.0979229866434775 alphas[1]
   + 0.9626216908551243 alphas[2] - 0.5708050594155364 alphas[3]
   - alphas[4] + f_var[110] = 0
 R1103: - 2156.17943612574 Fraction[110] + Q0_var[110] = 0
 R1104: Q0_var[110] + Lost_0_expr_110 = 164.7969259052417
 R1105: Q0_var[110] + Q_hat_adjusted_110 = 1116.293956777312
 R1106: Sold_0[110] - Q0_var[110] - Q1_var[110]
   + total_stock_second_period_110 = 0
 R1107: total_stock_second_period_110 + Lost_1_expr_110
   = 593.7018169584919
 R1108: Sold_1[110] + Left_1[110] - total_stock_second_period_110 = 0
 R1109: - 700 Sold_0[110] - 700 Sold_1[110] + 700 Lost_0[110]
   + 700 Lost_1[110] + 300 Left_1[110] + profits_vars[110] = 0
 Fraction_lower_bound_111: Fraction[111] >= 0
 Fraction_upper_bound_111: Fraction[111] <= 1
 R1112: - 1.252290367973626 alphas[0] + 0.5036110761409982 alphas[1]
   + 0.2388925425552696 alphas[2] + 0.3272267316514609 alphas[3]
   - alphas[4] + f_var[111] = 0
 R1113: - 2156.17943612574 Fraction[111] + Q0_var[111] = 0
 R1114: Q0_var[111] + Lost_0_expr_111 = 347.4935862583258
 R1115: Q0_var[111] + Q_hat_adjusted_111 = 1692.334778804114
 R1116: Sold_0[111] - Q0_var[111] - Q1_var[111]
   + total_stock_second_period_111 = 0
 R1117: total_stock_second_period_111 + Lost_1_expr_111
   = 899.6992617283089
 R1118: Sold_1[111] + Left_1[111] - total_stock_second_period_111 = 0
 R1119: - 700 Sold_0[111] - 700 Sold_1[111] + 700 Lost_0[111]
   + 700 Lost_1[111] + 300 Left_1[111] + profits_vars[111] = 0
 Fraction_lower_bound_112: Fraction[112] >= 0
 Fraction_upper_bound_112: Fraction[112] <= 1
 R1122: - 0.7324723239931097 alphas[0] - 1.653661811080881 alphas[1]
   + 0.4546748086002528 alphas[2] + 0.4424087857185023 alphas[3]
   - alphas[4] + f_var[112] = 0
 R1123: - 2156.17943612574 Fraction[112] + Q0_var[112] = 0
 R1124: Q0_var[112] + Lost_0_expr_112 = 325.4657758465807
 R1125: Q0_var[112] + Q_hat_adjusted_112 = 1541.308336039875
 R1126: Sold_0[112] - Q0_var[112] - Q1_var[112]
   + total_stock_second_period_112 = 0
 R1127: total_stock_second_period_112 + Lost_1_expr_112 = 1162.10442496777
 R1128: Sold_1[112] + Left_1[112] - total_stock_second_period_112 = 0
 R1129: - 700 Sold_0[112] - 700 Sold_1[112] + 700 Lost_0[112]
   + 700 Lost_1[112] + 300 Left_1[112] + profits_vars[112] = 0
 Fraction_lower_bound_113: Fraction[113] >= 0
 Fraction_upper_bound_113: Fraction[113] <= 1
 R1132: 0.6899037202024535 alphas[0] + 1.198957356027285 alphas[1]
   + 1.090825552899192 alphas[2] + 0.4086070377215792 alphas[3] - alphas[4]
   + f_var[113] = 0
 R1133: - 2156.17943612574 Fraction[113] + Q0_var[113] = 0
 R1134: Q0_var[113] + Lost_0_expr_113 = 226.4630016195126
 R1135: Q0_var[113] + Q_hat_adjusted_113 = 1074.499086035225
 R1136: Sold_0[113] - Q0_var[113] - Q1_var[113]
   + total_stock_second_period_113 = 0
 R1137: total_stock_second_period_113 + Lost_1_expr_113
   = 566.5965447154175
 R1138: Sold_1[113] + Left_1[113] - total_stock_second_period_113 = 0
 R1139: - 700 Sold_0[113] - 700 Sold_1[113] + 700 Lost_0[113]
   + 700 Lost_1[113] + 300 Left_1[113] + profits_vars[113] = 0
 Fraction_lower_bound_114: Fraction[114] >= 0
 Fraction_upper_bound_114: Fraction[114] <= 1
 R1142: - 1.136917202173371 alphas[0] + 1.323808364044422 alphas[1]
   + 0.1805904443469429 alphas[2] - 0.2521953361102858 alphas[3]
   - alphas[4] + f_var[114] = 0
 R1143: - 2156.17943612574 Fraction[114] + Q0_var[114] = 0
 R1144: Q0_var[114] + Lost_0_expr_114 = 304.8590739172243
 R1145: Q0_var[114] + Q_hat_adjusted_114 = 1701.961230651991
 R1146: Sold_0[114] - Q0_var[114] - Q1_var[114]
   + total_stock_second_period_114 = 0
 R1147: total_stock_second_period_114 + Lost_1_expr_114
   = 1093.668741310533
 R1148: Sold_1[114] + Left_1[114] - total_stock_second_period_114 = 0
 R1149: - 700 Sold_0[114] - 700 Sold_1[114] + 700 Lost_0[114]
   + 700 Lost_1[114] + 300 Left_1[114] + profits_vars[114] = 0
 Fraction_lower_bound_115: Fraction[115] >= 0
 Fraction_upper_bound_115: Fraction[115] <= 1
 R1152: 0.3394105879177239 alphas[0] + 0.4834554178304724 alphas[1]
   + 0.7659358419513157 alphas[2] - 1.233218589820831 alphas[3] - alphas[4]
   + f_var[115] = 0
 R1153: - 2156.17943612574 Fraction[115] + Q0_var[115] = 0
 R1154: Q0_var[115] + Lost_0_expr_115 = 194.0610952892391
 R1155: Q0_var[115] + Q_hat_adjusted_115 = 1259.625641609063
 R1156: Sold_0[115] - Q0_var[115] - Q1_var[115]
   + total_stock_second_period_115 = 0
 R1157: total_stock_second_period_115 + Lost_1_expr_115
   = 1023.130195168347
 R1158: Sold_1[115] + Left_1[115] - total_stock_second_period_115 = 0
 R1159: - 700 Sold_0[115] - 700 Sold_1[115] + 700 Lost_0[115]
   + 700 Lost_1[115] + 300 Left_1[115] + profits_vars[115] = 0
 Fraction_lower_bound_116: Fraction[116] >= 0
 Fraction_upper_bound_116: Fraction[116] <= 1
 R1162: - 1.37503530727505 alphas[0] - 0.3604669782669481 alphas[1]
   - 0.0838963360699979 alphas[2] - 1.392878149143411 alphas[3] - alphas[4]
   + f_var[116] = 0
 R1163: - 2156.17943612574 Fraction[116] + Q0_var[116] = 0
 R1164: Q0_var[116] + Lost_0_expr_116 = 414.618201065166
 R1165: Q0_var[116] + Q_hat_adjusted_116 = 1940.118072612957
 R1166: Sold_0[116] - Q0_var[116] - Q1_var[116]
   + total_stock_second_period_116 = 0
 R1167: total_stock_second_period_116 + Lost_1_expr_116
   = 1143.534004850062
 R1168: Sold_1[116] + Left_1[116] - total_stock_second_period_116 = 0
 R1169: - 700 Sold_0[116] - 700 Sold_1[116] + 700 Lost_0[116]
   + 700 Lost_1[116] + 300 Left_1[116] + profits_vars[116] = 0
 Fraction_lower_bound_117: Fraction[117] >= 0
 Fraction_upper_bound_117: Fraction[117] <= 1
 R1172: - 0.3142726837693007 alphas[0] - 0.5927156881375353 alphas[1]
   + 1.512112923999761 alphas[2] + 1.550091379814735 alphas[3] - alphas[4]
   + f_var[117] = 0
 R1173: - 2156.17943612574 Fraction[117] + Q0_var[117] = 0
 R1174: Q0_var[117] + Lost_0_expr_117 = 69.57894900517923
 R1175: Q0_var[117] + Q_hat_adjusted_117 = 705.1168156182084
 R1176: Sold_0[117] - Q0_var[117] - Q1_var[117]
   + total_stock_second_period_117 = 0
 R1177: total_stock_second_period_117 + Lost_1_expr_117
   = 460.3482183110156
 R1178: Sold_1[117] + Left_1[117] - total_stock_second_period_117 = 0
 R1179: - 700 Sold_0[117] - 700 Sold_1[117] + 700 Lost_0[117]
   + 700 Lost_1[117] + 300 Left_1[117] + profits_vars[117] = 0
 Fraction_lower_bound_118: Fraction[118] >= 0
 Fraction_upper_bound_118: Fraction[118] <= 1
 R1182: - 1.377271827991154 alphas[0] + 0.2998874093246034 alphas[1]
   - 1.343157594912136 alphas[2] + 1.119528957694624 alphas[3] - alphas[4]
   + f_var[118] = 0
 R1183: - 2156.17943612574 Fraction[118] + Q0_var[118] = 0
 R1184: Q0_var[118] + Lost_0_expr_118 = 379.8010388769554
 R1185: Q0_var[118] + Q_hat_adjusted_118 = 2716.397192537303
 R1186: Sold_0[118] - Q0_var[118] - Q1_var[118]
   + total_stock_second_period_118 = 0
 R1187: total_stock_second_period_118 + Lost_1_expr_118
   = 2493.894020546862
 R1188: Sold_1[118] + Left_1[118] - total_stock_second_period_118 = 0
 R1189: - 700 Sold_0[118] - 700 Sold_1[118] + 700 Lost_0[118]
   + 700 Lost_1[118] + 300 Left_1[118] + profits_vars[118] = 0
 Fraction_lower_bound_119: Fraction[119] >= 0
 Fraction_upper_bound_119: Fraction[119] <= 1
 R1192: - 0.707892272475324 alphas[0] - 1.751756582907006 alphas[1]
   - 1.048583655566302 alphas[2] + 1.352798984338308 alphas[3] - alphas[4]
   + f_var[119] = 0
 R1193: - 2156.17943612574 Fraction[119] + Q0_var[119] = 0
 R1194: Q0_var[119] + Lost_0_expr_119 = 594.3387977609357
 R1195: Q0_var[119] + Q_hat_adjusted_119 = 2667.228817256158
 R1196: Sold_0[119] - Q0_var[119] - Q1_var[119]
   + total_stock_second_period_119 = 0
 R1197: total_stock_second_period_119 + Lost_1_expr_119
   = 1863.788427215063
 R1198: Sold_1[119] + Left_1[119] - total_stock_second_period_119 = 0
 R1199: - 700 Sold_0[119] - 700 Sold_1[119] + 700 Lost_0[119]
   + 700 Lost_1[119] + 300 Left_1[119] + profits_vars[119] = 0
 Fraction_lower_bound_120: Fraction[120] >= 0
 Fraction_upper_bound_120: Fraction[120] <= 1
 R1202: - 0.8236611036617599 alphas[0] + 0.4544837257415876 alphas[1]
   - 0.1839337764070722 alphas[2] + 1.080160202661153 alphas[3] - alphas[4]
   + f_var[120] = 0
 R1203: - 2156.17943612574 Fraction[120] + Q0_var[120] = 0
 R1204: Q0_var[120] + Lost_0_expr_120 = 281.2113416298586
 R1205: Q0_var[120] + Q_hat_adjusted_120 = 1917.96739234261
 R1206: Sold_0[120] - Q0_var[120] - Q1_var[120]
   + total_stock_second_period_120 = 0
 R1207: total_stock_second_period_120 + Lost_1_expr_120 = 1494.35651349244
 R1208: Sold_1[120] + Left_1[120] - total_stock_second_period_120 = 0
 R1209: - 700 Sold_0[120] - 700 Sold_1[120] + 700 Lost_0[120]
   + 700 Lost_1[120] + 300 Left_1[120] + profits_vars[120] = 0
 Fraction_lower_bound_121: Fraction[121] >= 0
 Fraction_upper_bound_121: Fraction[121] <= 1
 R1212: - 0.0314248067295812 alphas[0] - 0.8047916051798187 alphas[1]
   - 1.164765870364171 alphas[2] + 0.5649580958861022 alphas[3] - alphas[4]
   + f_var[121] = 0
 R1213: - 2156.17943612574 Fraction[121] + Q0_var[121] = 0
 R1214: Q0_var[121] + Lost_0_expr_121 = 492.6667298467351
 R1215: Q0_var[121] + Q_hat_adjusted_121 = 2675.677475865679
 R1216: Sold_0[121] - Q0_var[121] - Q1_var[121]
   + total_stock_second_period_121 = 0
 R1217: total_stock_second_period_121 + Lost_1_expr_121
   = 2131.334263124195
 R1218: Sold_1[121] + Left_1[121] - total_stock_second_period_121 = 0
 R1219: - 700 Sold_0[121] - 700 Sold_1[121] + 700 Lost_0[121]
   + 700 Lost_1[121] + 300 Left_1[121] + profits_vars[121] = 0
 Fraction_lower_bound_122: Fraction[122] >= 0
 Fraction_upper_bound_122: Fraction[122] <= 1
 R1222: - 1.640306826599079 alphas[0] - 0.5191256882073663 alphas[1]
   - 1.527255514920235 alphas[2] + 0.0274007669044573 alphas[3] - alphas[4]
   + f_var[122] = 0
 R1223: - 2156.17943612574 Fraction[122] + Q0_var[122] = 0
 R1224: Q0_var[122] + Lost_0_expr_122 = 441.9225754434032
 R1225: Q0_var[122] + Q_hat_adjusted_122 = 2873.008890736827
 R1226: Sold_0[122] - Q0_var[122] - Q1_var[122]
   + total_stock_second_period_122 = 0
 R1227: total_stock_second_period_122 + Lost_1_expr_122
   = 2150.220862066671
 R1228: Sold_1[122] + Left_1[122] - total_stock_second_period_122 = 0
 R1229: - 700 Sold_0[122] - 700 Sold_1[122] + 700 Lost_0[122]
   + 700 Lost_1[122] + 300 Left_1[122] + profits_vars[122] = 0
 Fraction_lower_bound_123: Fraction[123] >= 0
 Fraction_upper_bound_123: Fraction[123] <= 1
 R1232: - 0.5361588322544704 alphas[0] - 1.117118485053687 alphas[1]
   + 1.265084292588115 alphas[2] - 0.385806321576956 alphas[3] - alphas[4]
   + f_var[123] = 0
 R1233: - 2156.17943612574 Fraction[123] + Q0_var[123] = 0
 R1234: Q0_var[123] + Lost_0_expr_123 = 132.9698882223018
 R1235: Q0_var[123] + Q_hat_adjusted_123 = 904.5444066731894
 R1236: Sold_0[123] - Q0_var[123] - Q1_var[123]
   + total_stock_second_period_123 = 0
 R1237: total_stock_second_period_123 + Lost_1_expr_123
   = 874.5734396807923
 R1238: Sold_1[123] + Left_1[123] - total_stock_second_period_123 = 0
 R1239: - 700 Sold_0[123] - 700 Sold_1[123] + 700 Lost_0[123]
   + 700 Lost_1[123] + 300 Left_1[123] + profits_vars[123] = 0
 Fraction_lower_bound_124: Fraction[124] >= 0
 Fraction_upper_bound_124: Fraction[124] <= 1
 R1242: 0.2426521181079669 alphas[0] - 1.673195941455682 alphas[1]
   - 0.4404297274481679 alphas[2] - 0.7309512906922638 alphas[3]
   - alphas[4] + f_var[124] = 0
 R1243: - 2156.17943612574 Fraction[124] + Q0_var[124] = 0
 R1244: Q0_var[124] + Lost_0_expr_124 = 237.7003103513032
 R1245: Q0_var[124] + Q_hat_adjusted_124 = 2052.718115986567
 R1246: Sold_0[124] - Q0_var[124] - Q1_var[124]
   + total_stock_second_period_124 = 0
 R1247: total_stock_second_period_124 + Lost_1_expr_124
   = 1781.278731126739
 R1248: Sold_1[124] + Left_1[124] - total_stock_second_period_124 = 0
 R1249: - 700 Sold_0[124] - 700 Sold_1[124] + 700 Lost_0[124]
   + 700 Lost_1[124] + 300 Left_1[124] + profits_vars[124] = 0
 Fraction_lower_bound_125: Fraction[125] >= 0
 Fraction_upper_bound_125: Fraction[125] <= 1
 R1252: - 0.403145346636591 alphas[0] - 1.378642038464481 alphas[1]
   - 1.643944654046798 alphas[2] - 1.26193872776804 alphas[3] - alphas[4]
   + f_var[125] = 0
 R1253: - 2156.17943612574 Fraction[125] + Q0_var[125] = 0
 R1254: Q0_var[125] + Lost_0_expr_125 = 607.3807953967404
 R1255: Q0_var[125] + Q_hat_adjusted_125 = 3053.171872949629
 R1256: Sold_0[125] - Q0_var[125] - Q1_var[125]
   + total_stock_second_period_125 = 0
 R1257: total_stock_second_period_125 + Lost_1_expr_125
   = 2201.016465399305
 R1258: Sold_1[125] + Left_1[125] - total_stock_second_period_125 = 0
 R1259: - 700 Sold_0[125] - 700 Sold_1[125] + 700 Lost_0[125]
   + 700 Lost_1[125] + 300 Left_1[125] + profits_vars[125] = 0
 Fraction_lower_bound_126: Fraction[126] >= 0
 Fraction_upper_bound_126: Fraction[126] <= 1
 R1262: 1.674295720714992 alphas[0] - 0.9518617054150537 alphas[1]
   - 1.61082219233507 alphas[2] - 1.594632997178369 alphas[3] - alphas[4]
   + f_var[126] = 0
 R1263: - 2156.17943612574 Fraction[126] + Q0_var[126] = 0
 R1264: Q0_var[126] + Lost_0_expr_126 = 612.072256724917
 R1265: Q0_var[126] + Q_hat_adjusted_126 = 3035.175828492218
 R1266: Sold_0[126] - Q0_var[126] - Q1_var[126]
   + total_stock_second_period_126 = 0
 R1267: total_stock_second_period_126 + Lost_1_expr_126
   = 2569.115550324982
 R1268: Sold_1[126] + Left_1[126] - total_stock_second_period_126 = 0
 R1269: - 700 Sold_0[126] - 700 Sold_1[126] + 700 Lost_0[126]
   + 700 Lost_1[126] + 300 Left_1[126] + profits_vars[126] = 0
 Fraction_lower_bound_127: Fraction[127] >= 0
 Fraction_upper_bound_127: Fraction[127] <= 1
 R1272: 0.6752644624489254 alphas[0] - 0.7258702017721045 alphas[1]
   - 0.3502590304475138 alphas[2] - 0.4980588702226368 alphas[3]
   - alphas[4] + f_var[127] = 0
 R1273: - 2156.17943612574 Fraction[127] + Q0_var[127] = 0
 R1274: Q0_var[127] + Lost_0_expr_127 = 499.9933882080895
 R1275: Q0_var[127] + Q_hat_adjusted_127 = 2163.821908538671
 R1276: Sold_0[127] - Q0_var[127] - Q1_var[127]
   + total_stock_second_period_127 = 0
 R1277: total_stock_second_period_127 + Lost_1_expr_127 = 1146.24535678492
 R1278: Sold_1[127] + Left_1[127] - total_stock_second_period_127 = 0
 R1279: - 700 Sold_0[127] - 700 Sold_1[127] + 700 Lost_0[127]
   + 700 Lost_1[127] + 300 Left_1[127] + profits_vars[127] = 0
 Fraction_lower_bound_128: Fraction[128] >= 0
 Fraction_upper_bound_128: Fraction[128] <= 1
 R1282: - 0.5934134824738325 alphas[0] + 0.5103560849131787 alphas[1]
   + 1.207725996123742 alphas[2] - 1.111529100303733 alphas[3] - alphas[4]
   + f_var[128] = 0
 R1283: - 2156.17943612574 Fraction[128] + Q0_var[128] = 0
 R1284: Q0_var[128] + Lost_0_expr_128 = 91.24704265603059
 R1285: Q0_var[128] + Q_hat_adjusted_128 = 913.9667098354633
 R1286: Sold_0[128] - Q0_var[128] - Q1_var[128]
   + total_stock_second_period_128 = 0
 R1287: total_stock_second_period_128 + Lost_1_expr_128
   = 616.4591849142245
 R1288: Sold_1[128] + Left_1[128] - total_stock_second_period_128 = 0
 R1289: - 700 Sold_0[128] - 700 Sold_1[128] + 700 Lost_0[128]
   + 700 Lost_1[128] + 300 Left_1[128] + profits_vars[128] = 0
 Fraction_lower_bound_129: Fraction[129] >= 0
 Fraction_upper_bound_129: Fraction[129] <= 1
 R1292: 0.7159401682684805 alphas[0] + 1.150407539958565 alphas[1]
   - 0.2842747050108949 alphas[2] - 0.88522512317734 alphas[3] - alphas[4]
   + f_var[129] = 0
 R1293: - 2156.17943612574 Fraction[129] + Q0_var[129] = 0
 R1294: Q0_var[129] + Lost_0_expr_129 = 343.9310519172101
 R1295: Q0_var[129] + Q_hat_adjusted_129 = 2021.844108778865
 R1296: Sold_0[129] - Q0_var[129] - Q1_var[129]
   + total_stock_second_period_129 = 0
 R1297: total_stock_second_period_129 + Lost_1_expr_129
   = 1565.309122799456
 R1298: Sold_1[129] + Left_1[129] - total_stock_second_period_129 = 0
 R1299: - 700 Sold_0[129] - 700 Sold_1[129] + 700 Lost_0[129]
   + 700 Lost_1[129] + 300 Left_1[129] + profits_vars[129] = 0
 Fraction_lower_bound_130: Fraction[130] >= 0
 Fraction_upper_bound_130: Fraction[130] <= 1
 R1302: - 0.4442633061325035 alphas[0] + 1.440250217161031 alphas[1]
   + 1.620595871774759 alphas[2] + 0.696304761221882 alphas[3] - alphas[4]
   + f_var[130] = 0
 R1303: - 2156.17943612574 Fraction[130] + Q0_var[130] = 0
 R1304: Q0_var[130] + Lost_0_expr_130 = 150.2829633311394
 R1305: Q0_var[130] + Q_hat_adjusted_130 = 698.3136226884636
 R1306: Sold_0[130] - Q0_var[130] - Q1_var[130]
   + total_stock_second_period_130 = 0
 R1307: total_stock_second_period_130 + Lost_1_expr_130
   = 430.9177596813311
 R1308: Sold_1[130] + Left_1[130] - total_stock_second_period_130 = 0
 R1309: - 700 Sold_0[130] - 700 Sold_1[130] + 700 Lost_0[130]
   + 700 Lost_1[130] + 300 Left_1[130] + profits_vars[130] = 0
 Fraction_lower_bound_131: Fraction[131] >= 0
 Fraction_upper_bound_131: Fraction[131] <= 1
 R1312: 0.2252682233775958 alphas[0] + 0.8293259337682335 alphas[1]
   + 0.9753553066184332 alphas[2] - 1.213470761177791 alphas[3] - alphas[4]
   + f_var[131] = 0
 R1313: - 2156.17943612574 Fraction[131] + Q0_var[131] = 0
 R1314: Q0_var[131] + Lost_0_expr_131 = 128.1786537167152
 R1315: Q0_var[131] + Q_hat_adjusted_131 = 1084.53022719564
 R1316: Sold_0[131] - Q0_var[131] - Q1_var[131]
   + total_stock_second_period_131 = 0
 R1317: total_stock_second_period_131 + Lost_1_expr_131
   = 959.5639272525543
 R1318: Sold_1[131] + Left_1[131] - total_stock_second_period_131 = 0
 R1319: - 700 Sold_0[131] - 700 Sold_1[131] + 700 Lost_0[131]
   + 700 Lost_1[131] + 300 Left_1[131] + profits_vars[131] = 0
 Fraction_lower_bound_132: Fraction[132] >= 0
 Fraction_upper_bound_132: Fraction[132] <= 1
 R1322: 1.262908391546832 alphas[0] + 0.1805256999078004 alphas[1]
   - 1.506696910105003 alphas[2] + 1.586281107921547 alphas[3] - alphas[4]
   + f_var[132] = 0
 R1323: - 2156.17943612574 Fraction[132] + Q0_var[132] = 0
 R1324: Q0_var[132] + Lost_0_expr_132 = 582.4663955143527
 R1325: Q0_var[132] + Q_hat_adjusted_132 = 2950.144709067446
 R1326: Sold_0[132] - Q0_var[132] - Q1_var[132]
   + total_stock_second_period_132 = 0
 R1327: total_stock_second_period_132 + Lost_1_expr_132
   = 2428.568912674506
 R1328: Sold_1[132] + Left_1[132] - total_stock_second_period_132 = 0
 R1329: - 700 Sold_0[132] - 700 Sold_1[132] + 700 Lost_0[132]
   + 700 Lost_1[132] + 300 Left_1[132] + profits_vars[132] = 0
 Fraction_lower_bound_133: Fraction[133] >= 0
 Fraction_upper_bound_133: Fraction[133] <= 1
 R1332: 0.6869128875992357 alphas[0] - 0.1252156936626845 alphas[1]
   + 0.5442177756784713 alphas[2] + 1.728787474240164 alphas[3] - alphas[4]
   + f_var[133] = 0
 R1333: - 2156.17943612574 Fraction[133] + Q0_var[133] = 0
 R1334: Q0_var[133] + Lost_0_expr_133 = 169.0792382893863
 R1335: Q0_var[133] + Q_hat_adjusted_133 = 1384.239833139389
 R1336: Sold_0[133] - Q0_var[133] - Q1_var[133]
   + total_stock_second_period_133 = 0
 R1337: total_stock_second_period_133 + Lost_1_expr_133
   = 1208.332902807657
 R1338: Sold_1[133] + Left_1[133] - total_stock_second_period_133 = 0
 R1339: - 700 Sold_0[133] - 700 Sold_1[133] + 700 Lost_0[133]
   + 700 Lost_1[133] + 300 Left_1[133] + profits_vars[133] = 0
 Fraction_lower_bound_134: Fraction[134] >= 0
 Fraction_upper_bound_134: Fraction[134] <= 1
 R1342: - 0.2742665123147165 alphas[0] - 0.980902620306796 alphas[1]
   - 0.4652049463737638 alphas[2] + 0.8099158202886336 alphas[3]
   - alphas[4] + f_var[134] = 0
 R1343: - 2156.17943612574 Fraction[134] + Q0_var[134] = 0
 R1344: Q0_var[134] + Lost_0_expr_134 = 396.4351079840676
 R1345: Q0_var[134] + Q_hat_adjusted_134 = 2170.276326583073
 R1346: Sold_0[134] - Q0_var[134] - Q1_var[134]
   + total_stock_second_period_134 = 0
 R1347: total_stock_second_period_134 + Lost_1_expr_134 = 1588.0688228323
 R1348: Sold_1[134] + Left_1[134] - total_stock_second_period_134 = 0
 R1349: - 700 Sold_0[134] - 700 Sold_1[134] + 700 Lost_0[134]
   + 700 Lost_1[134] + 300 Left_1[134] + profits_vars[134] = 0
 Fraction_lower_bound_135: Fraction[135] >= 0
 Fraction_upper_bound_135: Fraction[135] <= 1
 R1352: - 0.3482359029586286 alphas[0] - 1.613609236193848 alphas[1]
   + 0.7067387776475592 alphas[2] + 0.1767587351609935 alphas[3]
   - alphas[4] + f_var[135] = 0
 R1353: - 2156.17943612574 Fraction[135] + Q0_var[135] = 0
 R1354: Q0_var[135] + Lost_0_expr_135 = 431.1781818651522
 R1355: Q0_var[135] + Q_hat_adjusted_135 = 1449.175538686972
 R1356: Sold_0[135] - Q0_var[135] - Q1_var[135]
   + total_stock_second_period_135 = 0
 R1357: total_stock_second_period_135 + Lost_1_expr_135
   = 763.0687619328085
 R1358: Sold_1[135] + Left_1[135] - total_stock_second_period_135 = 0
 R1359: - 700 Sold_0[135] - 700 Sold_1[135] + 700 Lost_0[135]
   + 700 Lost_1[135] + 300 Left_1[135] + profits_vars[135] = 0
 Fraction_lower_bound_136: Fraction[136] >= 0
 Fraction_upper_bound_136: Fraction[136] <= 1
 R1362: - 0.2896928496662558 alphas[0] + 1.253889553019911 alphas[1]
   - 1.479486305351159 alphas[2] - 0.4157277886397545 alphas[3] - alphas[4]
   + f_var[136] = 0
 R1363: - 2156.17943612574 Fraction[136] + Q0_var[136] = 0
 R1364: Q0_var[136] + Lost_0_expr_136 = 627.6920260953062
 R1365: Q0_var[136] + Q_hat_adjusted_136 = 2961.904984769718
 R1366: Sold_0[136] - Q0_var[136] - Q1_var[136]
   + total_stock_second_period_136 = 0
 R1367: total_stock_second_period_136 + Lost_1_expr_136
   = 2366.903109825657
 R1368: Sold_1[136] + Left_1[136] - total_stock_second_period_136 = 0
 R1369: - 700 Sold_0[136] - 700 Sold_1[136] + 700 Lost_0[136]
   + 700 Lost_1[136] + 300 Left_1[136] + profits_vars[136] = 0
 Fraction_lower_bound_137: Fraction[137] >= 0
 Fraction_upper_bound_137: Fraction[137] <= 1
 R1372: - 0.5687448694341215 alphas[0] + 1.289046879258543 alphas[1]
   - 0.7607672000904783 alphas[2] + 1.387938521289609 alphas[3] - alphas[4]
   + f_var[137] = 0
 R1373: - 2156.17943612574 Fraction[137] + Q0_var[137] = 0
 R1374: Q0_var[137] + Lost_0_expr_137 = 409.6848311562389
 R1375: Q0_var[137] + Q_hat_adjusted_137 = 2366.221558170392
 R1376: Sold_0[137] - Q0_var[137] - Q1_var[137]
   + total_stock_second_period_137 = 0
 R1377: total_stock_second_period_137 + Lost_1_expr_137
   = 1910.576358528986
 R1378: Sold_1[137] + Left_1[137] - total_stock_second_period_137 = 0
 R1379: - 700 Sold_0[137] - 700 Sold_1[137] + 700 Lost_0[137]
   + 700 Lost_1[137] + 300 Left_1[137] + profits_vars[137] = 0
 Fraction_lower_bound_138: Fraction[138] >= 0
 Fraction_upper_bound_138: Fraction[138] <= 1
 R1382: - 0.5648618737227059 alphas[0] - 0.3559270144176457 alphas[1]
   + 0.5468053862247474 alphas[2] - 0.5321886830002595 alphas[3]
   - alphas[4] + f_var[138] = 0
 R1383: - 2156.17943612574 Fraction[138] + Q0_var[138] = 0
 R1384: Q0_var[138] + Lost_0_expr_138 = 180.812043148223
 R1385: Q0_var[138] + Q_hat_adjusted_138 = 1390.069874179621
 R1386: Sold_0[138] - Q0_var[138] - Q1_var[138]
   + total_stock_second_period_138 = 0
 R1387: total_stock_second_period_138 + Lost_1_expr_138
   = 1034.750589107668
 R1388: Sold_1[138] + Left_1[138] - total_stock_second_period_138 = 0
 R1389: - 700 Sold_0[138] - 700 Sold_1[138] + 700 Lost_0[138]
   + 700 Lost_1[138] + 300 Left_1[138] + profits_vars[138] = 0
 Fraction_lower_bound_139: Fraction[139] >= 0
 Fraction_upper_bound_139: Fraction[139] <= 1
 R1392: 0.2158937903137499 alphas[0] - 0.886552546715524 alphas[1]
   - 0.9679382646662977 alphas[2] - 0.9651475967517408 alphas[3]
   - alphas[4] + f_var[139] = 0
 R1393: - 2156.17943612574 Fraction[139] + Q0_var[139] = 0
 R1394: Q0_var[139] + Lost_0_expr_139 = 460.5228852169035
 R1395: Q0_var[139] + Q_hat_adjusted_139 = 2530.226584440141
 R1396: Sold_0[139] - Q0_var[139] - Q1_var[139]
   + total_stock_second_period_139 = 0
 R1397: total_stock_second_period_139 + Lost_1_expr_139 = 2065.32762358913
 R1398: Sold_1[139] + Left_1[139] - total_stock_second_period_139 = 0
 R1399: - 700 Sold_0[139] - 700 Sold_1[139] + 700 Lost_0[139]
   + 700 Lost_1[139] + 300 Left_1[139] + profits_vars[139] = 0
 Fraction_lower_bound_140: Fraction[140] >= 0
 Fraction_upper_bound_140: Fraction[140] <= 1
 R1402: - 1.429672142931519 alphas[0] - 1.236724825932385 alphas[1]
   + 1.27280950891321 alphas[2] + 1.107123875820909 alphas[3] - alphas[4]
   + f_var[140] = 0
 R1403: - 2156.17943612574 Fraction[140] + Q0_var[140] = 0
 R1404: Q0_var[140] + Lost_0_expr_140 = 311.7037429414744
 R1405: Q0_var[140] + Q_hat_adjusted_140 = 1014.89917218713
 R1406: Sold_0[140] - Q0_var[140] - Q1_var[140]
   + total_stock_second_period_140 = 0
 R1407: total_stock_second_period_140 + Lost_1_expr_140
   = 760.6952909757124
 R1408: Sold_1[140] + Left_1[140] - total_stock_second_period_140 = 0
 R1409: - 700 Sold_0[140] - 700 Sold_1[140] + 700 Lost_0[140]
   + 700 Lost_1[140] + 300 Left_1[140] + profits_vars[140] = 0
 Fraction_lower_bound_141: Fraction[141] >= 0
 Fraction_upper_bound_141: Fraction[141] <= 1
 R1412: 0.4418104248154132 alphas[0] - 1.535537589994194 alphas[1]
   + 0.341387690253571 alphas[2] + 0.3346481750391171 alphas[3] - alphas[4]
   + f_var[141] = 0
 R1413: - 2156.17943612574 Fraction[141] + Q0_var[141] = 0
 R1414: Q0_var[141] + Lost_0_expr_141 = 257.6424200299509
 R1415: Q0_var[141] + Q_hat_adjusted_141 = 1569.721879623604
 R1416: Sold_0[141] - Q0_var[141] - Q1_var[141]
   + total_stock_second_period_141 = 0
 R1417: total_stock_second_period_141 + Lost_1_expr_141
   = 1344.087351754543
 R1418: Sold_1[141] + Left_1[141] - total_stock_second_period_141 = 0
 R1419: - 700 Sold_0[141] - 700 Sold_1[141] + 700 Lost_0[141]
   + 700 Lost_1[141] + 300 Left_1[141] + profits_vars[141] = 0
 Fraction_lower_bound_142: Fraction[142] >= 0
 Fraction_upper_bound_142: Fraction[142] <= 1
 R1422: 0.2001626589564745 alphas[0] - 1.697734982775664 alphas[1]
   + 0.902236235634396 alphas[2] - 0.0178202739795726 alphas[3] - alphas[4]
   + f_var[142] = 0
 R1423: - 2156.17943612574 Fraction[142] + Q0_var[142] = 0
 R1424: Q0_var[142] + Lost_0_expr_142 = 158.7478957893452
 R1425: Q0_var[142] + Q_hat_adjusted_142 = 1150.626008461351
 R1426: Sold_0[142] - Q0_var[142] - Q1_var[142]
   + total_stock_second_period_142 = 0
 R1427: total_stock_second_period_142 + Lost_1_expr_142
   = 785.3959373659736
 R1428: Sold_1[142] + Left_1[142] - total_stock_second_period_142 = 0
 R1429: - 700 Sold_0[142] - 700 Sold_1[142] + 700 Lost_0[142]
   + 700 Lost_1[142] + 300 Left_1[142] + profits_vars[142] = 0
 Fraction_lower_bound_143: Fraction[143] >= 0
 Fraction_upper_bound_143: Fraction[143] <= 1
 R1432: - 1.413315688372661 alphas[0] + 0.2912144243363724 alphas[1]
   - 0.6138121963092124 alphas[2] + 0.2113485673683329 alphas[3]
   - alphas[4] + f_var[143] = 0
 R1433: - 2156.17943612574 Fraction[143] + Q0_var[143] = 0
 R1434: Q0_var[143] + Lost_0_expr_143 = 395.4554827517138
 R1435: Q0_var[143] + Q_hat_adjusted_143 = 2263.893712863909
 R1436: Sold_0[143] - Q0_var[143] - Q1_var[143]
   + total_stock_second_period_143 = 0
 R1437: total_stock_second_period_143 + Lost_1_expr_143
   = 2243.649026322146
 R1438: Sold_1[143] + Left_1[143] - total_stock_second_period_143 = 0
 R1439: - 700 Sold_0[143] - 700 Sold_1[143] + 700 Lost_0[143]
   + 700 Lost_1[143] + 300 Left_1[143] + profits_vars[143] = 0
 Fraction_lower_bound_144: Fraction[144] >= 0
 Fraction_upper_bound_144: Fraction[144] <= 1
 R1442: - 1.110015789282988 alphas[0] + 0.3575548497501934 alphas[1]
   + 1.291556520734449 alphas[2] - 0.9584722047024458 alphas[3] - alphas[4]
   + f_var[144] = 0
 R1443: - 2156.17943612574 Fraction[144] + Q0_var[144] = 0
 R1444: Q0_var[144] + Lost_0_expr_144 = 88.29395789537652
 R1445: Q0_var[144] + Q_hat_adjusted_144 = 857.6654982796396
 R1446: Sold_0[144] - Q0_var[144] - Q1_var[144]
   + total_stock_second_period_144 = 0
 R1447: total_stock_second_period_144 + Lost_1_expr_144 = 610.554720428368
 R1448: Sold_1[144] + Left_1[144] - total_stock_second_period_144 = 0
 R1449: - 700 Sold_0[144] - 700 Sold_1[144] + 700 Lost_0[144]
   + 700 Lost_1[144] + 300 Left_1[144] + profits_vars[144] = 0
 Fraction_lower_bound_145: Fraction[145] >= 0
 Fraction_upper_bound_145: Fraction[145] <= 1
 R1452: - 0.7480719114148066 alphas[0] + 1.14998802938484 alphas[1]
   + 0.3259590454839714 alphas[2] - 1.498613037866025 alphas[3] - alphas[4]
   + f_var[145] = 0
 R1453: - 2156.17943612574 Fraction[145] + Q0_var[145] = 0
 R1454: Q0_var[145] + Lost_0_expr_145 = 297.1912583119029
 R1455: Q0_var[145] + Q_hat_adjusted_145 = 1604.848150135557
 R1456: Sold_0[145] - Q0_var[145] - Q1_var[145]
   + total_stock_second_period_145 = 0
 R1457: total_stock_second_period_145 + Lost_1_expr_145
   = 1123.575342018475
 R1458: Sold_1[145] + Left_1[145] - total_stock_second_period_145 = 0
 R1459: - 700 Sold_0[145] - 700 Sold_1[145] + 700 Lost_0[145]
   + 700 Lost_1[145] + 300 Left_1[145] + profits_vars[145] = 0
 Fraction_lower_bound_146: Fraction[146] >= 0
 Fraction_upper_bound_146: Fraction[146] <= 1
 R1462: 1.387608546262039 alphas[0] - 0.6804969387014342 alphas[1]
   + 0.7205378953773932 alphas[2] - 0.9555250995806476 alphas[3]
   - alphas[4] + f_var[146] = 0
 R1463: - 2156.17943612574 Fraction[146] + Q0_var[146] = 0
 R1464: Q0_var[146] + Lost_0_expr_146 = 163.0980850681777
 R1465: Q0_var[146] + Q_hat_adjusted_146 = 1268.542701454647
 R1466: Sold_0[146] - Q0_var[146] - Q1_var[146]
   + total_stock_second_period_146 = 0
 R1467: total_stock_second_period_146 + Lost_1_expr_146
   = 1158.784798359883
 R1468: Sold_1[146] + Left_1[146] - total_stock_second_period_146 = 0
 R1469: - 700 Sold_0[146] - 700 Sold_1[146] + 700 Lost_0[146]
   + 700 Lost_1[146] + 300 Left_1[146] + profits_vars[146] = 0
 Fraction_lower_bound_147: Fraction[147] >= 0
 Fraction_upper_bound_147: Fraction[147] <= 1
 R1472: - 1.510816938929648 alphas[0] - 0.5844872045801222 alphas[1]
   - 0.0323696404877393 alphas[2] + 1.467862435308401 alphas[3] - alphas[4]
   + f_var[147] = 0
 R1473: - 2156.17943612574 Fraction[147] + Q0_var[147] = 0
 R1474: Q0_var[147] + Lost_0_expr_147 = 259.4600328125892
 R1475: Q0_var[147] + Q_hat_adjusted_147 = 1807.88937197056
 R1476: Sold_0[147] - Q0_var[147] - Q1_var[147]
   + total_stock_second_period_147 = 0
 R1477: total_stock_second_period_147 + Lost_1_expr_147
   = 1468.671003214039
 R1478: Sold_1[147] + Left_1[147] - total_stock_second_period_147 = 0
 R1479: - 700 Sold_0[147] - 700 Sold_1[147] + 700 Lost_0[147]
   + 700 Lost_1[147] + 300 Left_1[147] + profits_vars[147] = 0
 Fraction_lower_bound_148: Fraction[148] >= 0
 Fraction_upper_bound_148: Fraction[148] <= 1
 R1482: - 0.7846985408695304 alphas[0] - 1.284136629937247 alphas[1]
   + 0.4804396409357801 alphas[2] - 0.9756339632919433 alphas[3]
   - alphas[4] + f_var[148] = 0
 R1483: - 2156.17943612574 Fraction[148] + Q0_var[148] = 0
 R1484: Q0_var[148] + Lost_0_expr_148 = 308.8668891196473
 R1485: Q0_var[148] + Q_hat_adjusted_148 = 1514.36025102214
 R1486: Sold_0[148] - Q0_var[148] - Q1_var[148]
   + total_stock_second_period_148 = 0
 R1487: total_stock_second_period_148 + Lost_1_expr_148
   = 1335.992750757796
 R1488: Sold_1[148] + Left_1[148] - total_stock_second_period_148 = 0
 R1489: - 700 Sold_0[148] - 700 Sold_1[148] + 700 Lost_0[148]
   + 700 Lost_1[148] + 300 Left_1[148] + profits_vars[148] = 0
 Fraction_lower_bound_149: Fraction[149] >= 0
 Fraction_upper_bound_149: Fraction[149] <= 1
 R1492: - 1.791598349853963 alphas[0] + 1.322261023738915 alphas[1]
   - 0.6881681351236733 alphas[2] + 1.506169814334639 alphas[3] - alphas[4]
   + f_var[149] = 0
 R1493: - 2156.17943612574 Fraction[149] + Q0_var[149] = 0
 R1494: Q0_var[149] + Lost_0_expr_149 = 491.9489027829728
 R1495: Q0_var[149] + Q_hat_adjusted_149 = 2372.960838418242
 R1496: Sold_0[149] - Q0_var[149] - Q1_var[149]
   + total_stock_second_period_149 = 0
 R1497: total_stock_second_period_149 + Lost_1_expr_149
   = 1970.735136213621
 R1498: Sold_1[149] + Left_1[149] - total_stock_second_period_149 = 0
 R1499: - 700 Sold_0[149] - 700 Sold_1[149] + 700 Lost_0[149]
   + 700 Lost_1[149] + 300 Left_1[149] + profits_vars[149] = 0
 Fraction_lower_bound_150: Fraction[150] >= 0
 Fraction_upper_bound_150: Fraction[150] <= 1
 R1502: 1.213469256356462 alphas[0] - 0.0426778818883362 alphas[1]
   + 1.546802882124551 alphas[2] + 0.9368472836429583 alphas[3] - alphas[4]
   + f_var[150] = 0
 R1503: - 2156.17943612574 Fraction[150] + Q0_var[150] = 0
 R1504: Q0_var[150] + Lost_0_expr_150 = 56.84210518658441
 R1505: Q0_var[150] + Q_hat_adjusted_150 = 675.7036145536231
 R1506: Sold_0[150] - Q0_var[150] - Q1_var[150]
   + total_stock_second_period_150 = 0
 R1507: total_stock_second_period_150 + Lost_1_expr_150
   = 444.8153705408223
 R1508: Sold_1[150] + Left_1[150] - total_stock_second_period_150 = 0
 R1509: - 700 Sold_0[150] - 700 Sold_1[150] + 700 Lost_0[150]
   + 700 Lost_1[150] + 300 Left_1[150] + profits_vars[150] = 0
 Fraction_lower_bound_151: Fraction[151] >= 0
 Fraction_upper_bound_151: Fraction[151] <= 1
 R1512: - 1.329123791075281 alphas[0] - 0.3265752501557235 alphas[1]
   - 0.4507718202422053 alphas[2] + 1.215412644210837 alphas[3] - alphas[4]
   + f_var[151] = 0
 R1513: - 2156.17943612574 Fraction[151] + Q0_var[151] = 0
 R1514: Q0_var[151] + Lost_0_expr_151 = 593.3234035566596
 R1515: Q0_var[151] + Q_hat_adjusted_151 = 2287.447893994397
 R1516: Sold_0[151] - Q0_var[151] - Q1_var[151]
   + total_stock_second_period_151 = 0
 R1517: total_stock_second_period_151 + Lost_1_expr_151 = 1616.26382576109
 R1518: Sold_1[151] + Left_1[151] - total_stock_second_period_151 = 0
 R1519: - 700 Sold_0[151] - 700 Sold_1[151] + 700 Lost_0[151]
   + 700 Lost_1[151] + 300 Left_1[151] + profits_vars[151] = 0
 Fraction_lower_bound_152: Fraction[152] >= 0
 Fraction_upper_bound_152: Fraction[152] <= 1
 R1522: 1.167318967905235 alphas[0] + 0.8305005238017711 alphas[1]
   + 0.871577659502152 alphas[2] + 1.066094820298822 alphas[3] - alphas[4]
   + f_var[152] = 0
 R1523: - 2156.17943612574 Fraction[152] + Q0_var[152] = 0
 R1524: Q0_var[152] + Lost_0_expr_152 = 147.5218430662936
 R1525: Q0_var[152] + Q_hat_adjusted_152 = 1162.631974883199
 R1526: Sold_0[152] - Q0_var[152] - Q1_var[152]
   + total_stock_second_period_152 = 0
 R1527: total_stock_second_period_152 + Lost_1_expr_152
   = 1068.072225804329
 R1528: Sold_1[152] + Left_1[152] - total_stock_second_period_152 = 0
 R1529: - 700 Sold_0[152] - 700 Sold_1[152] + 700 Lost_0[152]
   + 700 Lost_1[152] + 300 Left_1[152] + profits_vars[152] = 0
 Fraction_lower_bound_153: Fraction[153] >= 0
 Fraction_upper_bound_153: Fraction[153] <= 1
 R1532: - 0.4355747603604417 alphas[0] + 1.077682980836419 alphas[1]
   + 0.7465665903794855 alphas[2] - 0.0458206356224078 alphas[3]
   - alphas[4] + f_var[153] = 0
 R1533: - 2156.17943612574 Fraction[153] + Q0_var[153] = 0
 R1534: Q0_var[153] + Lost_0_expr_153 = 438.273641190731
 R1535: Q0_var[153] + Q_hat_adjusted_153 = 1428.666140366679
 R1536: Sold_0[153] - Q0_var[153] - Q1_var[153]
   + total_stock_second_period_153 = 0
 R1537: total_stock_second_period_153 + Lost_1_expr_153
   = 907.8757321283105
 R1538: Sold_1[153] + Left_1[153] - total_stock_second_period_153 = 0
 R1539: - 700 Sold_0[153] - 700 Sold_1[153] + 700 Lost_0[153]
   + 700 Lost_1[153] + 300 Left_1[153] + profits_vars[153] = 0
 Fraction_lower_bound_154: Fraction[154] >= 0
 Fraction_upper_bound_154: Fraction[154] <= 1
 R1542: 1.304139090787864 alphas[0] - 1.275678976896877 alphas[1]
   - 0.9957812057957167 alphas[2] - 0.5277507954507116 alphas[3]
   - alphas[4] + f_var[154] = 0
 R1543: - 2156.17943612574 Fraction[154] + Q0_var[154] = 0
 R1544: Q0_var[154] + Lost_0_expr_154 = 341.4584729260217
 R1545: Q0_var[154] + Q_hat_adjusted_154 = 2471.491804119775
 R1546: Sold_0[154] - Q0_var[154] - Q1_var[154]
   + total_stock_second_period_154 = 0
 R1547: total_stock_second_period_154 + Lost_1_expr_154
   = 1949.136613191803
 R1548: Sold_1[154] + Left_1[154] - total_stock_second_period_154 = 0
 R1549: - 700 Sold_0[154] - 700 Sold_1[154] + 700 Lost_0[154]
   + 700 Lost_1[154] + 300 Left_1[154] + profits_vars[154] = 0
 Fraction_lower_bound_155: Fraction[155] >= 0
 Fraction_upper_bound_155: Fraction[155] <= 1
 R1552: - 1.257949402378488 alphas[0] + 1.454226398035204 alphas[1]
   - 1.504878453114251 alphas[2] + 0.1491159668576848 alphas[3] - alphas[4]
   + f_var[155] = 0
 R1553: - 2156.17943612574 Fraction[155] + Q0_var[155] = 0
 R1554: Q0_var[155] + Lost_0_expr_155 = 537.4245899083228
 R1555: Q0_var[155] + Q_hat_adjusted_155 = 2920.092283222589
 R1556: Sold_0[155] - Q0_var[155] - Q1_var[155]
   + total_stock_second_period_155 = 0
 R1557: total_stock_second_period_155 + Lost_1_expr_155
   = 2253.130409996704
 R1558: Sold_1[155] + Left_1[155] - total_stock_second_period_155 = 0
 R1559: - 700 Sold_0[155] - 700 Sold_1[155] + 700 Lost_0[155]
   + 700 Lost_1[155] + 300 Left_1[155] + profits_vars[155] = 0
 Fraction_lower_bound_156: Fraction[156] >= 0
 Fraction_upper_bound_156: Fraction[156] <= 1
 R1562: - 1.113995792940287 alphas[0] + 0.049867775876664 alphas[1]
   + 0.5897487672568423 alphas[2] - 1.495834025561088 alphas[3] - alphas[4]
   + f_var[156] = 0
 R1563: - 2156.17943612574 Fraction[156] + Q0_var[156] = 0
 R1564: Q0_var[156] + Lost_0_expr_156 = 202.4215749083674
 R1565: Q0_var[156] + Q_hat_adjusted_156 = 1376.728237691078
 R1566: Sold_0[156] - Q0_var[156] - Q1_var[156]
   + total_stock_second_period_156 = 0
 R1567: total_stock_second_period_156 + Lost_1_expr_156
   = 1396.343674834774
 R1568: Sold_1[156] + Left_1[156] - total_stock_second_period_156 = 0
 R1569: - 700 Sold_0[156] - 700 Sold_1[156] + 700 Lost_0[156]
   + 700 Lost_1[156] + 300 Left_1[156] + profits_vars[156] = 0
 Fraction_lower_bound_157: Fraction[157] >= 0
 Fraction_upper_bound_157: Fraction[157] <= 1
 R1572: - 0.2712091782994784 alphas[0] + 1.258955364066543 alphas[1]
   - 1.082807922933489 alphas[2] + 0.9223198232223189 alphas[3] - alphas[4]
   + f_var[157] = 0
 R1573: - 2156.17943612574 Fraction[157] + Q0_var[157] = 0
 R1574: Q0_var[157] + Lost_0_expr_157 = 698.8419203542333
 R1575: Q0_var[157] + Q_hat_adjusted_157 = 2755.983696185321
 R1576: Sold_0[157] - Q0_var[157] - Q1_var[157]
   + total_stock_second_period_157 = 0
 R1577: total_stock_second_period_157 + Lost_1_expr_157
   = 2051.825627528687
 R1578: Sold_1[157] + Left_1[157] - total_stock_second_period_157 = 0
 R1579: - 700 Sold_0[157] - 700 Sold_1[157] + 700 Lost_0[157]
   + 700 Lost_1[157] + 300 Left_1[157] + profits_vars[157] = 0
 Fraction_lower_bound_158: Fraction[158] >= 0
 Fraction_upper_bound_158: Fraction[158] <= 1
 R1582: 0.3016347133891701 alphas[0] + 0.0960865572067635 alphas[1]
   + 1.286493645311348 alphas[2] - 0.8351459168539561 alphas[3] - alphas[4]
   + f_var[158] = 0
 R1583: - 2156.17943612574 Fraction[158] + Q0_var[158] = 0
 R1584: Q0_var[158] + Lost_0_expr_158 = 75.97277971462395
 R1585: Q0_var[158] + Q_hat_adjusted_158 = 853.5387001870622
 R1586: Sold_0[158] - Q0_var[158] - Q1_var[158]
   + total_stock_second_period_158 = 0
 R1587: total_stock_second_period_158 + Lost_1_expr_158
   = 585.3635202546136
 R1588: Sold_1[158] + Left_1[158] - total_stock_second_period_158 = 0
 R1589: - 700 Sold_0[158] - 700 Sold_1[158] + 700 Lost_0[158]
   + 700 Lost_1[158] + 300 Left_1[158] + profits_vars[158] = 0
 Fraction_lower_bound_159: Fraction[159] >= 0
 Fraction_upper_bound_159: Fraction[159] <= 1
 R1592: 1.497494604084463 alphas[0] - 1.685930212891279 alphas[1]
   - 0.4507545513729133 alphas[2] - 1.194949510070406 alphas[3] - alphas[4]
   + f_var[159] = 0
 R1593: - 2156.17943612574 Fraction[159] + Q0_var[159] = 0
 R1594: Q0_var[159] + Lost_0_expr_159 = 524.5361651542147
 R1595: Q0_var[159] + Q_hat_adjusted_159 = 2243.302428230091
 R1596: Sold_0[159] - Q0_var[159] - Q1_var[159]
   + total_stock_second_period_159 = 0
 R1597: total_stock_second_period_159 + Lost_1_expr_159
   = 1528.725534777989
 R1598: Sold_1[159] + Left_1[159] - total_stock_second_period_159 = 0
 R1599: - 700 Sold_0[159] - 700 Sold_1[159] + 700 Lost_0[159]
   + 700 Lost_1[159] + 300 Left_1[159] + profits_vars[159] = 0
 Fraction_lower_bound_160: Fraction[160] >= 0
 Fraction_upper_bound_160: Fraction[160] <= 1
 R1602: - 0.7252179007846519 alphas[0] + 0.2097491025304893 alphas[1]
   - 0.8344329450952989 alphas[2] - 0.1592689183486682 alphas[3]
   - alphas[4] + f_var[160] = 0
 R1603: - 2156.17943612574 Fraction[160] + Q0_var[160] = 0
 R1604: Q0_var[160] + Lost_0_expr_160 = 444.0140374725378
 R1605: Q0_var[160] + Q_hat_adjusted_160 = 2434.965923806086
 R1606: Sold_0[160] - Q0_var[160] - Q1_var[160]
   + total_stock_second_period_160 = 0
 R1607: total_stock_second_period_160 + Lost_1_expr_160
   = 2133.806755197715
 R1608: Sold_1[160] + Left_1[160] - total_stock_second_period_160 = 0
 R1609: - 700 Sold_0[160] - 700 Sold_1[160] + 700 Lost_0[160]
   + 700 Lost_1[160] + 300 Left_1[160] + profits_vars[160] = 0
 Fraction_lower_bound_161: Fraction[161] >= 0
 Fraction_upper_bound_161: Fraction[161] <= 1
 R1612: 0.1376209385239737 alphas[0] - 1.267309386430352 alphas[1]
   + 1.116086015780944 alphas[2] + 0.196506657344287 alphas[3] - alphas[4]
   + f_var[161] = 0
 R1613: - 2156.17943612574 Fraction[161] + Q0_var[161] = 0
 R1614: Q0_var[161] + Lost_0_expr_161 = 239.3424487672534
 R1615: Q0_var[161] + Q_hat_adjusted_161 = 1066.406263621076
 R1616: Sold_0[161] - Q0_var[161] - Q1_var[161]
   + total_stock_second_period_161 = 0
 R1617: total_stock_second_period_161 + Lost_1_expr_161
   = 865.4934553755439
 R1618: Sold_1[161] + Left_1[161] - total_stock_second_period_161 = 0
 R1619: - 700 Sold_0[161] - 700 Sold_1[161] + 700 Lost_0[161]
   + 700 Lost_1[161] + 300 Left_1[161] + profits_vars[161] = 0
 Fraction_lower_bound_162: Fraction[162] >= 0
 Fraction_upper_bound_162: Fraction[162] <= 1
 R1622: - 0.8123445680010999 alphas[0] + 1.253906307072309 alphas[1]
   + 0.2309772700195105 alphas[2] - 0.2739519944381401 alphas[3]
   - alphas[4] + f_var[162] = 0
 R1623: - 2156.17943612574 Fraction[162] + Q0_var[162] = 0
 R1624: Q0_var[162] + Lost_0_expr_162 = 403.676960995897
 R1625: Q0_var[162] + Q_hat_adjusted_162 = 1733.408006262261
 R1626: Sold_0[162] - Q0_var[162] - Q1_var[162]
   + total_stock_second_period_162 = 0
 R1627: total_stock_second_period_162 + Lost_1_expr_162 = 1421.91640672116
 R1628: Sold_1[162] + Left_1[162] - total_stock_second_period_162 = 0
 R1629: - 700 Sold_0[162] - 700 Sold_1[162] + 700 Lost_0[162]
   + 700 Lost_1[162] + 300 Left_1[162] + profits_vars[162] = 0
 Fraction_lower_bound_163: Fraction[163] >= 0
 Fraction_upper_bound_163: Fraction[163] <= 1
 R1632: - 1.322954684798988 alphas[0] + 0.7293020295324172 alphas[1]
   - 1.298826659584869 alphas[2] + 0.2886499671680338 alphas[3] - alphas[4]
   + f_var[163] = 0
 R1633: - 2156.17943612574 Fraction[163] + Q0_var[163] = 0
 R1634: Q0_var[163] + Lost_0_expr_163 = 499.8746860815196
 R1635: Q0_var[163] + Q_hat_adjusted_163 = 2765.322888030292
 R1636: Sold_0[163] - Q0_var[163] - Q1_var[163]
   + total_stock_second_period_163 = 0
 R1637: total_stock_second_period_163 + Lost_1_expr_163
   = 2180.088476356245
 R1638: Sold_1[163] + Left_1[163] - total_stock_second_period_163 = 0
 R1639: - 700 Sold_0[163] - 700 Sold_1[163] + 700 Lost_0[163]
   + 700 Lost_1[163] + 300 Left_1[163] + profits_vars[163] = 0
 Fraction_lower_bound_164: Fraction[164] >= 0
 Fraction_upper_bound_164: Fraction[164] <= 1
 R1642: - 1.709075608773801 alphas[0] + 0.277613225700043 alphas[1]
   + 1.287845360993722 alphas[2] - 0.1361690726958736 alphas[3] - alphas[4]
   + f_var[164] = 0
 R1643: - 2156.17943612574 Fraction[164] + Q0_var[164] = 0
 R1644: Q0_var[164] + Lost_0_expr_164 = 178.869011957668
 R1645: Q0_var[164] + Q_hat_adjusted_164 = 915.9679572068762
 R1646: Sold_0[164] - Q0_var[164] - Q1_var[164]
   + total_stock_second_period_164 = 0
 R1647: total_stock_second_period_164 + Lost_1_expr_164
   = 491.3409414359294
 R1648: Sold_1[164] + Left_1[164] - total_stock_second_period_164 = 0
 R1649: - 700 Sold_0[164] - 700 Sold_1[164] + 700 Lost_0[164]
   + 700 Lost_1[164] + 300 Left_1[164] + profits_vars[164] = 0
 Fraction_lower_bound_165: Fraction[165] >= 0
 Fraction_upper_bound_165: Fraction[165] <= 1
 R1652: - 1.285527548803559 alphas[0] + 0.291178819056588 alphas[1]
   + 1.568045751219007 alphas[2] + 0.3285685418006594 alphas[3] - alphas[4]
   + f_var[165] = 0
 R1653: - 2156.17943612574 Fraction[165] + Q0_var[165] = 0
 R1654: Q0_var[165] + Lost_0_expr_165 = 120.0639925704784
 R1655: Q0_var[165] + Q_hat_adjusted_165 = 705.139180236577
 R1656: Sold_0[165] - Q0_var[165] - Q1_var[165]
   + total_stock_second_period_165 = 0
 R1657: total_stock_second_period_165 + Lost_1_expr_165
   = 809.0733215218597
 R1658: Sold_1[165] + Left_1[165] - total_stock_second_period_165 = 0
 R1659: - 700 Sold_0[165] - 700 Sold_1[165] + 700 Lost_0[165]
   + 700 Lost_1[165] + 300 Left_1[165] + profits_vars[165] = 0
 Fraction_lower_bound_166: Fraction[166] >= 0
 Fraction_upper_bound_166: Fraction[166] <= 1
 R1662: 1.700755901932181 alphas[0] - 0.6343162667560965 alphas[1]
   - 0.3085066169936591 alphas[2] + 0.383910335073444 alphas[3] - alphas[4]
   + f_var[166] = 0
 R1663: - 2156.17943612574 Fraction[166] + Q0_var[166] = 0
 R1664: Q0_var[166] + Lost_0_expr_166 = 445.6641819954741
 R1665: Q0_var[166] + Q_hat_adjusted_166 = 2102.484607326232
 R1666: Sold_0[166] - Q0_var[166] - Q1_var[166]
   + total_stock_second_period_166 = 0
 R1667: total_stock_second_period_166 + Lost_1_expr_166
   = 1745.544693587992
 R1668: Sold_1[166] + Left_1[166] - total_stock_second_period_166 = 0
 R1669: - 700 Sold_0[166] - 700 Sold_1[166] + 700 Lost_0[166]
   + 700 Lost_1[166] + 300 Left_1[166] + profits_vars[166] = 0
 Fraction_lower_bound_167: Fraction[167] >= 0
 Fraction_upper_bound_167: Fraction[167] <= 1
 R1672: 0.4686409914239532 alphas[0] + 0.4789351493149442 alphas[1]
   + 0.1997418162065616 alphas[2] - 0.6072685322866761 alphas[3]
   - alphas[4] + f_var[167] = 0
 R1673: - 2156.17943612574 Fraction[167] + Q0_var[167] = 0
 R1674: Q0_var[167] + Lost_0_expr_167 = 531.4675974688647
 R1675: Q0_var[167] + Q_hat_adjusted_167 = 1835.207499188137
 R1676: Sold_0[167] - Q0_var[167] - Q1_var[167]
   + total_stock_second_period_167 = 0
 R1677: total_stock_second_period_167 + Lost_1_expr_167
   = 1323.871842602416
 R1678: Sold_1[167] + Left_1[167] - total_stock_second_period_167 = 0
 R1679: - 700 Sold_0[167] - 700 Sold_1[167] + 700 Lost_0[167]
   + 700 Lost_1[167] + 300 Left_1[167] + profits_vars[167] = 0
 Fraction_lower_bound_168: Fraction[168] >= 0
 Fraction_upper_bound_168: Fraction[168] <= 1
 R1682: - 0.8404174860254137 alphas[0] - 0.7787557319727034 alphas[1]
   - 0.990646877426085 alphas[2] + 1.417918426750563 alphas[3] - alphas[4]
   + f_var[168] = 0
 R1683: - 2156.17943612574 Fraction[168] + Q0_var[168] = 0
 R1684: Q0_var[168] + Lost_0_expr_168 = 565.2566982574215
 R1685: Q0_var[168] + Q_hat_adjusted_168 = 2611.826286472608
 R1686: Sold_0[168] - Q0_var[168] - Q1_var[168]
   + total_stock_second_period_168 = 0
 R1687: total_stock_second_period_168 + Lost_1_expr_168
   = 1953.105752867165
 R1688: Sold_1[168] + Left_1[168] - total_stock_second_period_168 = 0
 R1689: - 700 Sold_0[168] - 700 Sold_1[168] + 700 Lost_0[168]
   + 700 Lost_1[168] + 300 Left_1[168] + profits_vars[168] = 0
 Fraction_lower_bound_169: Fraction[169] >= 0
 Fraction_upper_bound_169: Fraction[169] <= 1
 R1692: 1.134994302107033 alphas[0] - 0.5245926994494621 alphas[1]
   - 1.399616250062716 alphas[2] + 1.170218935793283 alphas[3] - alphas[4]
   + f_var[169] = 0
 R1693: - 2156.17943612574 Fraction[169] + Q0_var[169] = 0
 R1694: Q0_var[169] + Lost_0_expr_169 = 610.4619027649469
 R1695: Q0_var[169] + Q_hat_adjusted_169 = 2900.1967808902
 R1696: Sold_0[169] - Q0_var[169] - Q1_var[169]
   + total_stock_second_period_169 = 0
 R1697: total_stock_second_period_169 + Lost_1_expr_169
   = 1717.135387924426
 R1698: Sold_1[169] + Left_1[169] - total_stock_second_period_169 = 0
 R1699: - 700 Sold_0[169] - 700 Sold_1[169] + 700 Lost_0[169]
   + 700 Lost_1[169] + 300 Left_1[169] + profits_vars[169] = 0
 Fraction_lower_bound_170: Fraction[170] >= 0
 Fraction_upper_bound_170: Fraction[170] <= 1
 R1702: - 0.1011642194332471 alphas[0] + 0.2933973611501826 alphas[1]
   + 0.6467726838170753 alphas[2] - 1.13113895703116 alphas[3] - alphas[4]
   + f_var[170] = 0
 R1703: - 2156.17943612574 Fraction[170] + Q0_var[170] = 0
 R1704: Q0_var[170] + Lost_0_expr_170 = 264.7188384078374
 R1705: Q0_var[170] + Q_hat_adjusted_170 = 1380.553042066033
 R1706: Sold_0[170] - Q0_var[170] - Q1_var[170]
   + total_stock_second_period_170 = 0
 R1707: total_stock_second_period_170 + Lost_1_expr_170
   = 867.3899397526766
 R1708: Sold_1[170] + Left_1[170] - total_stock_second_period_170 = 0
 R1709: - 700 Sold_0[170] - 700 Sold_1[170] + 700 Lost_0[170]
   + 700 Lost_1[170] + 300 Left_1[170] + profits_vars[170] = 0
 Fraction_lower_bound_171: Fraction[171] >= 0
 Fraction_upper_bound_171: Fraction[171] <= 1
 R1712: 1.549957798125153 alphas[0] + 0.1823024342626471 alphas[1]
   + 0.3542200239923465 alphas[2] - 0.8618252057199925 alphas[3]
   - alphas[4] + f_var[171] = 0
 R1713: - 2156.17943612574 Fraction[171] + Q0_var[171] = 0
 R1714: Q0_var[171] + Lost_0_expr_171 = 275.8069251597127
 R1715: Q0_var[171] + Q_hat_adjusted_171 = 1573.206085667717
 R1716: Sold_0[171] - Q0_var[171] - Q1_var[171]
   + total_stock_second_period_171 = 0
 R1717: total_stock_second_period_171 + Lost_1_expr_171
   = 1440.908027352427
 R1718: Sold_1[171] + Left_1[171] - total_stock_second_period_171 = 0
 R1719: - 700 Sold_0[171] - 700 Sold_1[171] + 700 Lost_0[171]
   + 700 Lost_1[171] + 300 Left_1[171] + profits_vars[171] = 0
 Fraction_lower_bound_172: Fraction[172] >= 0
 Fraction_upper_bound_172: Fraction[172] <= 1
 R1722: 1.034635901833461 alphas[0] - 0.4405559671813808 alphas[1]
   + 0.0334202136689771 alphas[2] + 0.7839626419740909 alphas[3]
   - alphas[4] + f_var[172] = 0
 R1723: - 2156.17943612574 Fraction[172] + Q0_var[172] = 0
 R1724: Q0_var[172] + Lost_0_expr_172 = 245.5623510792364
 R1725: Q0_var[172] + Q_hat_adjusted_172 = 1757.250322935503
 R1726: Sold_0[172] - Q0_var[172] - Q1_var[172]
   + total_stock_second_period_172 = 0
 R1727: total_stock_second_period_172 + Lost_1_expr_172
   = 1222.357150859487
 R1728: Sold_1[172] + Left_1[172] - total_stock_second_period_172 = 0
 R1729: - 700 Sold_0[172] - 700 Sold_1[172] + 700 Lost_0[172]
   + 700 Lost_1[172] + 300 Left_1[172] + profits_vars[172] = 0
 Fraction_lower_bound_173: Fraction[173] >= 0
 Fraction_upper_bound_173: Fraction[173] <= 1
 R1732: 1.676671064534021 alphas[0] + 1.415010477999216 alphas[1]
   - 0.3005285151082764 alphas[2] + 0.4618958659087449 alphas[3]
   - alphas[4] + f_var[173] = 0
 R1733: - 2156.17943612574 Fraction[173] + Q0_var[173] = 0
 R1734: Q0_var[173] + Lost_0_expr_173 = 400.02188960258
 R1735: Q0_var[173] + Q_hat_adjusted_173 = 2068.140455143558
 R1736: Sold_0[173] - Q0_var[173] - Q1_var[173]
   + total_stock_second_period_173 = 0
 R1737: total_stock_second_period_173 + Lost_1_expr_173
   = 1181.695170898573
 R1738: Sold_1[173] + Left_1[173] - total_stock_second_period_173 = 0
 R1739: - 700 Sold_0[173] - 700 Sold_1[173] + 700 Lost_0[173]
   + 700 Lost_1[173] + 300 Left_1[173] + profits_vars[173] = 0
 Fraction_lower_bound_174: Fraction[174] >= 0
 Fraction_upper_bound_174: Fraction[174] <= 1
 R1742: - 1.065805475284954 alphas[0] - 1.148992393373951 alphas[1]
   - 1.597594818700661 alphas[2] - 1.382660872268745 alphas[3] - alphas[4]
   + f_var[174] = 0
 R1743: - 2156.17943612574 Fraction[174] + Q0_var[174] = 0
 R1744: Q0_var[174] + Lost_0_expr_174 = 574.7071782693196
 R1745: Q0_var[174] + Q_hat_adjusted_174 = 3002.813368417542
 R1746: Sold_0[174] - Q0_var[174] - Q1_var[174]
   + total_stock_second_period_174 = 0
 R1747: total_stock_second_period_174 + Lost_1_expr_174
   = 2186.666447987837
 R1748: Sold_1[174] + Left_1[174] - total_stock_second_period_174 = 0
 R1749: - 700 Sold_0[174] - 700 Sold_1[174] + 700 Lost_0[174]
   + 700 Lost_1[174] + 300 Left_1[174] + profits_vars[174] = 0
 Fraction_lower_bound_175: Fraction[175] >= 0
 Fraction_upper_bound_175: Fraction[175] <= 1
 R1752: 0.9499810185159868 alphas[0] - 0.5731020137707138 alphas[1]
   - 0.658403880771953 alphas[2] - 1.019807430355133 alphas[3] - alphas[4]
   + f_var[175] = 0
 R1753: - 2156.17943612574 Fraction[175] + Q0_var[175] = 0
 R1754: Q0_var[175] + Lost_0_expr_175 = 427.174132175296
 R1755: Q0_var[175] + Q_hat_adjusted_175 = 2312.524484687963
 R1756: Sold_0[175] - Q0_var[175] - Q1_var[175]
   + total_stock_second_period_175 = 0
 R1757: total_stock_second_period_175 + Lost_1_expr_175
   = 1494.365643430667
 R1758: Sold_1[175] + Left_1[175] - total_stock_second_period_175 = 0
 R1759: - 700 Sold_0[175] - 700 Sold_1[175] + 700 Lost_0[175]
   + 700 Lost_1[175] + 300 Left_1[175] + profits_vars[175] = 0
 Fraction_lower_bound_176: Fraction[176] >= 0
 Fraction_upper_bound_176: Fraction[176] <= 1
 R1762: 0.5203873291553879 alphas[0] - 0.8216124353486733 alphas[1]
   + 0.3504688462097112 alphas[2] - 0.3350000241840412 alphas[3]
   - alphas[4] + f_var[176] = 0
 R1763: - 2156.17943612574 Fraction[176] + Q0_var[176] = 0
 R1764: Q0_var[176] + Lost_0_expr_176 = 355.7967346945577
 R1765: Q0_var[176] + Q_hat_adjusted_176 = 1626.898995000203
 R1766: Sold_0[176] - Q0_var[176] - Q1_var[176]
   + total_stock_second_period_176 = 0
 R1767: total_stock_second_period_176 + Lost_1_expr_176 = 1419.34095129156
 R1768: Sold_1[176] + Left_1[176] - total_stock_second_period_176 = 0
 R1769: - 700 Sold_0[176] - 700 Sold_1[176] + 700 Lost_0[176]
   + 700 Lost_1[176] + 300 Left_1[176] + profits_vars[176] = 0
 Fraction_lower_bound_177: Fraction[177] >= 0
 Fraction_upper_bound_177: Fraction[177] <= 1
 R1772: - 1.541238004828862 alphas[0] - 0.176084879536981 alphas[1]
   + 0.7627637352496004 alphas[2] + 0.9855200740040403 alphas[3]
   - alphas[4] + f_var[177] = 0
 R1773: - 2156.17943612574 Fraction[177] + Q0_var[177] = 0
 R1774: Q0_var[177] + Lost_0_expr_177 = 213.996161354379
 R1775: Q0_var[177] + Q_hat_adjusted_177 = 1274.345630807487
 R1776: Sold_0[177] - Q0_var[177] - Q1_var[177]
   + total_stock_second_period_177 = 0
 R1777: total_stock_second_period_177 + Lost_1_expr_177
   = 1071.755748224121
 R1778: Sold_1[177] + Left_1[177] - total_stock_second_period_177 = 0
 R1779: - 700 Sold_0[177] - 700 Sold_1[177] + 700 Lost_0[177]
   + 700 Lost_1[177] + 300 Left_1[177] + profits_vars[177] = 0
 Fraction_lower_bound_178: Fraction[178] >= 0
 Fraction_upper_bound_178: Fraction[178] <= 1
 R1782: - 0.7499321918537235 alphas[0] + 1.277211251012474 alphas[1]
   - 1.469009636445691 alphas[2] + 1.581276643745936 alphas[3] - alphas[4]
   + f_var[178] = 0
 R1783: - 2156.17943612574 Fraction[178] + Q0_var[178] = 0
 R1784: Q0_var[178] + Lost_0_expr_178 = 695.6388548663745
 R1785: Q0_var[178] + Q_hat_adjusted_178 = 2998.855927184264
 R1786: Sold_0[178] - Q0_var[178] - Q1_var[178]
   + total_stock_second_period_178 = 0
 R1787: total_stock_second_period_178 + Lost_1_expr_178
   = 2372.978510954937
 R1788: Sold_1[178] + Left_1[178] - total_stock_second_period_178 = 0
 R1789: - 700 Sold_0[178] - 700 Sold_1[178] + 700 Lost_0[178]
   + 700 Lost_1[178] + 300 Left_1[178] + profits_vars[178] = 0
 Fraction_lower_bound_179: Fraction[179] >= 0
 Fraction_upper_bound_179: Fraction[179] <= 1
 R1792: 1.629556686894299 alphas[0] + 0.273377607476587 alphas[1]
   + 1.183051521997983 alphas[2] + 0.2804211671426795 alphas[3] - alphas[4]
   + f_var[179] = 0
 R1793: - 2156.17943612574 Fraction[179] + Q0_var[179] = 0
 R1794: Q0_var[179] + Lost_0_expr_179 = 274.0257614515539
 R1795: Q0_var[179] + Q_hat_adjusted_179 = 1046.736341512798
 R1796: Sold_0[179] - Q0_var[179] - Q1_var[179]
   + total_stock_second_period_179 = 0
 R1797: total_stock_second_period_179 + Lost_1_expr_179
   = 843.5877060852167
 R1798: Sold_1[179] + Left_1[179] - total_stock_second_period_179 = 0
 R1799: - 700 Sold_0[179] - 700 Sold_1[179] + 700 Lost_0[179]
   + 700 Lost_1[179] + 300 Left_1[179] + profits_vars[179] = 0
 Fraction_lower_bound_180: Fraction[180] >= 0
 Fraction_upper_bound_180: Fraction[180] <= 1
 R1802: 1.159531317211736 alphas[0] + 0.2722258321798704 alphas[1]
   - 0.733584113181329 alphas[2] - 0.5309013108310032 alphas[3] - alphas[4]
   + f_var[180] = 0
 R1803: - 2156.17943612574 Fraction[180] + Q0_var[180] = 0
 R1804: Q0_var[180] + Lost_0_expr_180 = 329.2425263950671
 R1805: Q0_var[180] + Q_hat_adjusted_180 = 2297.369779758823
 R1806: Sold_0[180] - Q0_var[180] - Q1_var[180]
   + total_stock_second_period_180 = 0
 R1807: total_stock_second_period_180 + Lost_1_expr_180
   = 1800.426786142918
 R1808: Sold_1[180] + Left_1[180] - total_stock_second_period_180 = 0
 R1809: - 700 Sold_0[180] - 700 Sold_1[180] + 700 Lost_0[180]
   + 700 Lost_1[180] + 300 Left_1[180] + profits_vars[180] = 0
 Fraction_lower_bound_181: Fraction[181] >= 0
 Fraction_upper_bound_181: Fraction[181] <= 1
 R1812: - 0.4565148747543057 alphas[0] + 0.5596182706083196 alphas[1]
   - 1.405973662967072 alphas[2] - 1.054776265592766 alphas[3] - alphas[4]
   + f_var[181] = 0
 R1813: - 2156.17943612574 Fraction[181] + Q0_var[181] = 0
 R1814: Q0_var[181] + Lost_0_expr_181 = 578.6144911696799
 R1815: Q0_var[181] + Q_hat_adjusted_181 = 2883.795096215577
 R1816: Sold_0[181] - Q0_var[181] - Q1_var[181]
   + total_stock_second_period_181 = 0
 R1817: total_stock_second_period_181 + Lost_1_expr_181
   = 2038.985575954113
 R1818: Sold_1[181] + Left_1[181] - total_stock_second_period_181 = 0
 R1819: - 700 Sold_0[181] - 700 Sold_1[181] + 700 Lost_0[181]
   + 700 Lost_1[181] + 300 Left_1[181] + profits_vars[181] = 0
 Fraction_lower_bound_182: Fraction[182] >= 0
 Fraction_upper_bound_182: Fraction[182] <= 1
 R1822: - 0.2999645068123288 alphas[0] + 1.551640596787007 alphas[1]
   - 0.5513329136289097 alphas[2] - 1.136149137690028 alphas[3] - alphas[4]
   + f_var[182] = 0
 R1823: - 2156.17943612574 Fraction[182] + Q0_var[182] = 0
 R1824: Q0_var[182] + Lost_0_expr_182 = 407.1374655694316
 R1825: Q0_var[182] + Q_hat_adjusted_182 = 2231.764907539547
 R1826: Sold_0[182] - Q0_var[182] - Q1_var[182]
   + total_stock_second_period_182 = 0
 R1827: total_stock_second_period_182 + Lost_1_expr_182
   = 1614.874399577545
 R1828: Sold_1[182] + Left_1[182] - total_stock_second_period_182 = 0
 R1829: - 700 Sold_0[182] - 700 Sold_1[182] + 700 Lost_0[182]
   + 700 Lost_1[182] + 300 Left_1[182] + profits_vars[182] = 0
 Fraction_lower_bound_183: Fraction[183] >= 0
 Fraction_upper_bound_183: Fraction[183] <= 1
 R1832: 0.9005634895662304 alphas[0] - 0.8587989703181245 alphas[1]
   + 0.2406402987720384 alphas[2] + 1.61985153235083 alphas[3] - alphas[4]
   + f_var[183] = 0
 R1833: - 2156.17943612574 Fraction[183] + Q0_var[183] = 0
 R1834: Q0_var[183] + Lost_0_expr_183 = 289.1019003866597
 R1835: Q0_var[183] + Q_hat_adjusted_183 = 1653.774694261622
 R1836: Sold_0[183] - Q0_var[183] - Q1_var[183]
   + total_stock_second_period_183 = 0
 R1837: total_stock_second_period_183 + Lost_1_expr_183 = 1371.20119352674
 R1838: Sold_1[183] + Left_1[183] - total_stock_second_period_183 = 0
 R1839: - 700 Sold_0[183] - 700 Sold_1[183] + 700 Lost_0[183]
   + 700 Lost_1[183] + 300 Left_1[183] + profits_vars[183] = 0
 Fraction_lower_bound_184: Fraction[184] >= 0
 Fraction_upper_bound_184: Fraction[184] <= 1
 R1842: - 1.562934755091577 alphas[0] + 1.279571761091889 alphas[1]
   + 0.9750943698473263 alphas[2] + 0.9093292346062538 alphas[3]
   - alphas[4] + f_var[184] = 0
 R1843: - 2156.17943612574 Fraction[184] + Q0_var[184] = 0
 R1844: Q0_var[184] + Lost_0_expr_184 = 208.8372286312708
 R1845: Q0_var[184] + Q_hat_adjusted_184 = 1136.423667944526
 R1846: Sold_0[184] - Q0_var[184] - Q1_var[184]
   + total_stock_second_period_184 = 0
 R1847: total_stock_second_period_184 + Lost_1_expr_184
   = 992.1224068965528
 R1848: Sold_1[184] + Left_1[184] - total_stock_second_period_184 = 0
 R1849: - 700 Sold_0[184] - 700 Sold_1[184] + 700 Lost_0[184]
   + 700 Lost_1[184] + 300 Left_1[184] + profits_vars[184] = 0
 Fraction_lower_bound_185: Fraction[185] >= 0
 Fraction_upper_bound_185: Fraction[185] <= 1
 R1852: - 0.4299367711305477 alphas[0] - 0.4125442900793438 alphas[1]
   + 0.4228221774582746 alphas[2] + 0.5902125345784049 alphas[3]
   - alphas[4] + f_var[185] = 0
 R1853: - 2156.17943612574 Fraction[185] + Q0_var[185] = 0
 R1854: Q0_var[185] + Lost_0_expr_185 = 243.4325675892941
 R1855: Q0_var[185] + Q_hat_adjusted_185 = 1508.930497328717
 R1856: Sold_0[185] - Q0_var[185] - Q1_var[185]
   + total_stock_second_period_185 = 0
 R1857: total_stock_second_period_185 + Lost_1_expr_185
   = 1122.379430706891
 R1858: Sold_1[185] + Left_1[185] - total_stock_second_period_185 = 0
 R1859: - 700 Sold_0[185] - 700 Sold_1[185] + 700 Lost_0[185]
   + 700 Lost_1[185] + 300 Left_1[185] + profits_vars[185] = 0
 Fraction_lower_bound_186: Fraction[186] >= 0
 Fraction_upper_bound_186: Fraction[186] <= 1
 R1862: - 0.1528037594296232 alphas[0] - 0.7428042507169214 alphas[1]
   - 0.6896799991194503 alphas[2] + 1.095630500140895 alphas[3] - alphas[4]
   + f_var[186] = 0
 R1863: - 2156.17943612574 Fraction[186] + Q0_var[186] = 0
 R1864: Q0_var[186] + Lost_0_expr_186 = 523.8994973547306
 R1865: Q0_var[186] + Q_hat_adjusted_186 = 2394.419425754879
 R1866: Sold_0[186] - Q0_var[186] - Q1_var[186]
   + total_stock_second_period_186 = 0
 R1867: total_stock_second_period_186 + Lost_1_expr_186
   = 1822.958237429521
 R1868: Sold_1[186] + Left_1[186] - total_stock_second_period_186 = 0
 R1869: - 700 Sold_0[186] - 700 Sold_1[186] + 700 Lost_0[186]
   + 700 Lost_1[186] + 300 Left_1[186] + profits_vars[186] = 0
 Fraction_lower_bound_187: Fraction[187] >= 0
 Fraction_upper_bound_187: Fraction[187] <= 1
 R1872: - 0.3448296890228081 alphas[0] - 0.5095958601187965 alphas[1]
   - 0.5017340692225171 alphas[2] - 0.9690450999307966 alphas[3]
   - alphas[4] + f_var[187] = 0
 R1873: - 2156.17943612574 Fraction[187] + Q0_var[187] = 0
 R1874: Q0_var[187] + Lost_0_expr_187 = 383.8936940510458
 R1875: Q0_var[187] + Q_hat_adjusted_187 = 2185.396186524623
 R1876: Sold_0[187] - Q0_var[187] - Q1_var[187]
   + total_stock_second_period_187 = 0
 R1877: total_stock_second_period_187 + Lost_1_expr_187
   = 1826.742474944972
 R1878: Sold_1[187] + Left_1[187] - total_stock_second_period_187 = 0
 R1879: - 700 Sold_0[187] - 700 Sold_1[187] + 700 Lost_0[187]
   + 700 Lost_1[187] + 300 Left_1[187] + profits_vars[187] = 0
 Fraction_lower_bound_188: Fraction[188] >= 0
 Fraction_upper_bound_188: Fraction[188] <= 1
 R1882: - 0.8408826003477005 alphas[0] - 1.61497693554815 alphas[1]
   - 1.414434343703182 alphas[2] - 0.8496854714890645 alphas[3] - alphas[4]
   + f_var[188] = 0
 R1883: - 2156.17943612574 Fraction[188] + Q0_var[188] = 0
 R1884: Q0_var[188] + Lost_0_expr_188 = 498.6107403156558
 R1885: Q0_var[188] + Q_hat_adjusted_188 = 2837.829791433073
 R1886: Sold_0[188] - Q0_var[188] - Q1_var[188]
   + total_stock_second_period_188 = 0
 R1887: total_stock_second_period_188 + Lost_1_expr_188
   = 2360.409567194891
 R1888: Sold_1[188] + Left_1[188] - total_stock_second_period_188 = 0
 R1889: - 700 Sold_0[188] - 700 Sold_1[188] + 700 Lost_0[188]
   + 700 Lost_1[188] + 300 Left_1[188] + profits_vars[188] = 0
 Fraction_lower_bound_189: Fraction[189] >= 0
 Fraction_upper_bound_189: Fraction[189] <= 1
 R1892: 0.6385760517795859 alphas[0] + 1.301676603083843 alphas[1]
   - 1.214105658854354 alphas[2] - 0.00722552304052074 alphas[3]
   - alphas[4] + f_var[189] = 0
 R1893: - 2156.17943612574 Fraction[189] + Q0_var[189] = 0
 R1894: Q0_var[189] + Lost_0_expr_189 = 742.0075371204716
 R1895: Q0_var[189] + Q_hat_adjusted_189 = 2866.947508935742
 R1896: Sold_0[189] - Q0_var[189] - Q1_var[189]
   + total_stock_second_period_189 = 0
 R1897: total_stock_second_period_189 + Lost_1_expr_189
   = 1889.239071403888
 R1898: Sold_1[189] + Left_1[189] - total_stock_second_period_189 = 0
 R1899: - 700 Sold_0[189] - 700 Sold_1[189] + 700 Lost_0[189]
   + 700 Lost_1[189] + 300 Left_1[189] + profits_vars[189] = 0
 Fraction_lower_bound_190: Fraction[190] >= 0
 Fraction_upper_bound_190: Fraction[190] <= 1
 R1902: 0.3333419929587784 alphas[0] - 1.301532604895745 alphas[1]
   - 1.047883652860609 alphas[2] - 1.573920602553447 alphas[3] - alphas[4]
   + f_var[190] = 0
 R1903: - 2156.17943612574 Fraction[190] + Q0_var[190] = 0
 R1904: Q0_var[190] + Lost_0_expr_190 = 493.9450427889806
 R1905: Q0_var[190] + Q_hat_adjusted_190 = 2602.37148601303
 R1906: Sold_0[190] - Q0_var[190] - Q1_var[190]
   + total_stock_second_period_190 = 0
 R1907: total_stock_second_period_190 + Lost_1_expr_190
   = 2029.368961379122
 R1908: Sold_1[190] + Left_1[190] - total_stock_second_period_190 = 0
 R1909: - 700 Sold_0[190] - 700 Sold_1[190] + 700 Lost_0[190]
   + 700 Lost_1[190] + 300 Left_1[190] + profits_vars[190] = 0
 Fraction_lower_bound_191: Fraction[191] >= 0
 Fraction_upper_bound_191: Fraction[191] <= 1
 R1912: 0.999797640339152 alphas[0] + 1.554230927618383 alphas[1]
   - 1.360266362664474 alphas[2] + 1.245793564278825 alphas[3] - alphas[4]
   + f_var[191] = 0
 R1913: - 2156.17943612574 Fraction[191] + Q0_var[191] = 0
 R1914: Q0_var[191] + Lost_0_expr_191 = 432.8494975223789
 R1915: Q0_var[191] + Q_hat_adjusted_191 = 2761.283789110517
 R1916: Sold_0[191] - Q0_var[191] - Q1_var[191]
   + total_stock_second_period_191 = 0
 R1917: total_stock_second_period_191 + Lost_1_expr_191
   = 2423.534447115342
 R1918: Sold_1[191] + Left_1[191] - total_stock_second_period_191 = 0
 R1919: - 700 Sold_0[191] - 700 Sold_1[191] + 700 Lost_0[191]
   + 700 Lost_1[191] + 300 Left_1[191] + profits_vars[191] = 0
 Fraction_lower_bound_192: Fraction[192] >= 0
 Fraction_upper_bound_192: Fraction[192] <= 1
 R1922: 1.083471045745988 alphas[0] - 0.169248081451756 alphas[1]
   + 0.7215624879446487 alphas[2] - 1.294006385378832 alphas[3] - alphas[4]
   + f_var[192] = 0
 R1923: - 2156.17943612574 Fraction[192] + Q0_var[192] = 0
 R1924: Q0_var[192] + Lost_0_expr_192 = 119.0798743351343
 R1925: Q0_var[192] + Q_hat_adjusted_192 = 1239.643865785276
 R1926: Sold_0[192] - Q0_var[192] - Q1_var[192]
   + total_stock_second_period_192 = 0
 R1927: total_stock_second_period_192 + Lost_1_expr_192
   = 1250.971345856744
 R1928: Sold_1[192] + Left_1[192] - total_stock_second_period_192 = 0
 R1929: - 700 Sold_0[192] - 700 Sold_1[192] + 700 Lost_0[192]
   + 700 Lost_1[192] + 300 Left_1[192] + profits_vars[192] = 0
 Fraction_lower_bound_193: Fraction[193] >= 0
 Fraction_upper_bound_193: Fraction[193] <= 1
 R1932: - 1.598873890434575 alphas[0] + 0.2760766692227348 alphas[1]
   + 0.416108357650625 alphas[2] + 1.338550502609757 alphas[3] - alphas[4]
   + f_var[193] = 0
 R1933: - 2156.17943612574 Fraction[193] + Q0_var[193] = 0
 R1934: Q0_var[193] + Lost_0_expr_193 = 264.6647972575785
 R1935: Q0_var[193] + Q_hat_adjusted_193 = 1526.788706925732
 R1936: Sold_0[193] - Q0_var[193] - Q1_var[193]
   + total_stock_second_period_193 = 0
 R1937: total_stock_second_period_193 + Lost_1_expr_193
   = 1311.615246472861
 R1938: Sold_1[193] + Left_1[193] - total_stock_second_period_193 = 0
 R1939: - 700 Sold_0[193] - 700 Sold_1[193] + 700 Lost_0[193]
   + 700 Lost_1[193] + 300 Left_1[193] + profits_vars[193] = 0
 Fraction_lower_bound_194: Fraction[194] >= 0
 Fraction_upper_bound_194: Fraction[194] <= 1
 R1942: - 0.8742404072346355 alphas[0] - 0.1326708279550108 alphas[1]
   + 0.3821155128078634 alphas[2] + 1.181514247753837 alphas[3] - alphas[4]
   + f_var[194] = 0
 R1943: - 2156.17943612574 Fraction[194] + Q0_var[194] = 0
 R1944: Q0_var[194] + Lost_0_expr_194 = 230.2915183824858
 R1945: Q0_var[194] + Q_hat_adjusted_194 = 1526.294557152136
 R1946: Sold_0[194] - Q0_var[194] - Q1_var[194]
   + total_stock_second_period_194 = 0
 R1947: total_stock_second_period_194 + Lost_1_expr_194
   = 1066.820532836279
 R1948: Sold_1[194] + Left_1[194] - total_stock_second_period_194 = 0
 R1949: - 700 Sold_0[194] - 700 Sold_1[194] + 700 Lost_0[194]
   + 700 Lost_1[194] + 300 Left_1[194] + profits_vars[194] = 0
 Fraction_lower_bound_195: Fraction[195] >= 0
 Fraction_upper_bound_195: Fraction[195] <= 1
 R1952: 0.00701624845542044 alphas[0] + 0.4094756501413954 alphas[1]
   - 0.2097282821863953 alphas[2] - 0.6102020752071102 alphas[3]
   - alphas[4] + f_var[195] = 0
 R1953: - 2156.17943612574 Fraction[195] + Q0_var[195] = 0
 R1954: Q0_var[195] + Lost_0_expr_195 = 528.0286649174883
 R1955: Q0_var[195] + Q_hat_adjusted_195 = 2092.685688203473
 R1956: Sold_0[195] - Q0_var[195] - Q1_var[195]
   + total_stock_second_period_195 = 0
 R1957: total_stock_second_period_195 + Lost_1_expr_195
   = 1783.145862078497
 R1958: Sold_1[195] + Left_1[195] - total_stock_second_period_195 = 0
 R1959: - 700 Sold_0[195] - 700 Sold_1[195] + 700 Lost_0[195]
   + 700 Lost_1[195] + 300 Left_1[195] + profits_vars[195] = 0
 Fraction_lower_bound_196: Fraction[196] >= 0
 Fraction_upper_bound_196: Fraction[196] <= 1
 R1962: 0.9376340425241169 alphas[0] + 1.0042711081263 alphas[1]
   - 0.5629780638552726 alphas[2] + 1.361722057887049 alphas[3] - alphas[4]
   + f_var[196] = 0
 R1963: - 2156.17943612574 Fraction[196] + Q0_var[196] = 0
 R1964: Q0_var[196] + Lost_0_expr_196 = 545.0957351606476
 R1965: Q0_var[196] + Q_hat_adjusted_196 = 2327.665281649608
 R1966: Sold_0[196] - Q0_var[196] - Q1_var[196]
   + total_stock_second_period_196 = 0
 R1967: total_stock_second_period_196 + Lost_1_expr_196
   = 1555.276832110032
 R1968: Sold_1[196] + Left_1[196] - total_stock_second_period_196 = 0
 R1969: - 700 Sold_0[196] - 700 Sold_1[196] + 700 Lost_0[196]
   + 700 Lost_1[196] + 300 Left_1[196] + profits_vars[196] = 0
 Fraction_lower_bound_197: Fraction[197] >= 0
 Fraction_upper_bound_197: Fraction[197] <= 1
 R1972: 0.842317094293812 alphas[0] + 1.588539670717234 alphas[1]
   + 0.6875395958673856 alphas[2] - 0.1719713428001418 alphas[3]
   - alphas[4] + f_var[197] = 0
 R1973: - 2156.17943612574 Fraction[197] + Q0_var[197] = 0
 R1974: Q0_var[197] + Lost_0_expr_197 = 128.2293070037987
 R1975: Q0_var[197] + Q_hat_adjusted_197 = 1267.122637728748
 R1976: Sold_0[197] - Q0_var[197] - Q1_var[197]
   + total_stock_second_period_197 = 0
 R1977: total_stock_second_period_197 + Lost_1_expr_197
   = 1160.458602694817
 R1978: Sold_1[197] + Left_1[197] - total_stock_second_period_197 = 0
 R1979: - 700 Sold_0[197] - 700 Sold_1[197] + 700 Lost_0[197]
   + 700 Lost_1[197] + 300 Left_1[197] + profits_vars[197] = 0
 Fraction_lower_bound_198: Fraction[198] >= 0
 Fraction_upper_bound_198: Fraction[198] <= 1
 R1982: 1.536898887939632 alphas[0] - 0.1121072423211425 alphas[1]
   + 1.563567804589155 alphas[2] + 0.3174775412871053 alphas[3] - alphas[4]
   + f_var[198] = 0
 R1983: - 2156.17943612574 Fraction[198] + Q0_var[198] = 0
 R1984: Q0_var[198] + Lost_0_expr_198 = 147.5668046372598
 R1985: Q0_var[198] + Q_hat_adjusted_198 = 733.7614011277424
 R1986: Sold_0[198] - Q0_var[198] - Q1_var[198]
   + total_stock_second_period_198 = 0
 R1987: total_stock_second_period_198 + Lost_1_expr_198
   = 451.8371744690056
 R1988: Sold_1[198] + Left_1[198] - total_stock_second_period_198 = 0
 R1989: - 700 Sold_0[198] - 700 Sold_1[198] + 700 Lost_0[198]
   + 700 Lost_1[198] + 300 Left_1[198] + profits_vars[198] = 0
 Fraction_lower_bound_199: Fraction[199] >= 0
 Fraction_upper_bound_199: Fraction[199] <= 1
 R1992: 0.205286564285414 alphas[0] - 1.218412274194177 alphas[1]
   + 0.3187590057714125 alphas[2] - 0.7186210701835251 alphas[3]
   - alphas[4] + f_var[199] = 0
 R1993: - 2156.17943612574 Fraction[199] + Q0_var[199] = 0
 R1994: Q0_var[199] + Lost_0_expr_199 = 149.8046591276339
 R1995: Q0_var[199] + Q_hat_adjusted_199 = 1514.842141723668
 R1996: Sold_0[199] - Q0_var[199] - Q1_var[199]
   + total_stock_second_period_199 = 0
 R1997: total_stock_second_period_199 + Lost_1_expr_199
   = 1188.538643362749
 R1998: Sold_1[199] + Left_1[199] - total_stock_second_period_199 = 0
 R1999: - 700 Sold_0[199] - 700 Sold_1[199] + 700 Lost_0[199]
   + 700 Lost_1[199] + 300 Left_1[199] + profits_vars[199] = 0
 Fraction_lower_bound_200: Fraction[200] >= 0
 Fraction_upper_bound_200: Fraction[200] <= 1
 R2002: 0.6391035968581396 alphas[0] + 0.3818166527822581 alphas[1]
   + 0.6160447704546954 alphas[2] - 1.297446828587608 alphas[3] - alphas[4]
   + f_var[200] = 0
 R2003: - 2156.17943612574 Fraction[200] + Q0_var[200] = 0
 R2004: Q0_var[200] + Lost_0_expr_200 = 294.7966953176393
 R2005: Q0_var[200] + Q_hat_adjusted_200 = 1419.365115996303
 R2006: Sold_0[200] - Q0_var[200] - Q1_var[200]
   + total_stock_second_period_200 = 0
 R2007: total_stock_second_period_200 + Lost_1_expr_200
   = 1052.020125747054
 R2008: Sold_1[200] + Left_1[200] - total_stock_second_period_200 = 0
 R2009: - 700 Sold_0[200] - 700 Sold_1[200] + 700 Lost_0[200]
   + 700 Lost_1[200] + 300 Left_1[200] + profits_vars[200] = 0
 Fraction_lower_bound_201: Fraction[201] >= 0
 Fraction_upper_bound_201: Fraction[201] <= 1
 R2012: - 0.7213782989891874 alphas[0] + 0.8942056452899156 alphas[1]
   - 1.461008324550221 alphas[2] - 0.8577720027834056 alphas[3] - alphas[4]
   + f_var[201] = 0
 R2013: - 2156.17943612574 Fraction[201] + Q0_var[201] = 0
 R2014: Q0_var[201] + Lost_0_expr_201 = 499.3142297197068
 R2015: Q0_var[201] + Q_hat_adjusted_201 = 2867.818159654931
 R2016: Sold_0[201] - Q0_var[201] - Q1_var[201]
   + total_stock_second_period_201 = 0
 R2017: total_stock_second_period_201 + Lost_1_expr_201
   = 2317.277212541023
 R2018: Sold_1[201] + Left_1[201] - total_stock_second_period_201 = 0
 R2019: - 700 Sold_0[201] - 700 Sold_1[201] + 700 Lost_0[201]
   + 700 Lost_1[201] + 300 Left_1[201] + profits_vars[201] = 0
 Fraction_lower_bound_202: Fraction[202] >= 0
 Fraction_upper_bound_202: Fraction[202] <= 1
 R2022: 0.4057595732938267 alphas[0] + 1.37926196811121 alphas[1]
   - 1.284265895177422 alphas[2] - 1.126068066447736 alphas[3] - alphas[4]
   + f_var[202] = 0
 R2023: - 2156.17943612574 Fraction[202] + Q0_var[202] = 0
 R2024: Q0_var[202] + Lost_0_expr_202 = 728.4740636010729
 R2025: Q0_var[202] + Q_hat_adjusted_202 = 2902.759629794297
 R2026: Sold_0[202] - Q0_var[202] - Q1_var[202]
   + total_stock_second_period_202 = 0
 R2027: total_stock_second_period_202 + Lost_1_expr_202
   = 2008.803261169839
 R2028: Sold_1[202] + Left_1[202] - total_stock_second_period_202 = 0
 R2029: - 700 Sold_0[202] - 700 Sold_1[202] + 700 Lost_0[202]
   + 700 Lost_1[202] + 300 Left_1[202] + profits_vars[202] = 0
 Fraction_lower_bound_203: Fraction[203] >= 0
 Fraction_upper_bound_203: Fraction[203] <= 1
 R2032: 1.106783274783661 alphas[0] + 1.362975762540062 alphas[1]
   - 1.192634749986386 alphas[2] + 1.619281627688208 alphas[3] - alphas[4]
   + f_var[203] = 0
 R2033: - 2156.17943612574 Fraction[203] + Q0_var[203] = 0
 R2034: Q0_var[203] + Lost_0_expr_203 = 544.6209260811635
 R2035: Q0_var[203] + Q_hat_adjusted_203 = 2726.68599534733
 R2036: Sold_0[203] - Q0_var[203] - Q1_var[203]
   + total_stock_second_period_203 = 0
 R2037: total_stock_second_period_203 + Lost_1_expr_203
   = 2126.074526386978
 R2038: Sold_1[203] + Left_1[203] - total_stock_second_period_203 = 0
 R2039: - 700 Sold_0[203] - 700 Sold_1[203] + 700 Lost_0[203]
   + 700 Lost_1[203] + 300 Left_1[203] + profits_vars[203] = 0
 Fraction_lower_bound_204: Fraction[204] >= 0
 Fraction_upper_bound_204: Fraction[204] <= 1
 R2042: 1.654888593396291 alphas[0] + 0.8992063398575416 alphas[1]
   - 0.5130936953392855 alphas[2] + 0.6873519894963785 alphas[3]
   - alphas[4] + f_var[204] = 0
 R2043: - 2156.17943612574 Fraction[204] + Q0_var[204] = 0
 R2044: Q0_var[204] + Lost_0_expr_204 = 562.5581607230132
 R2045: Q0_var[204] + Q_hat_adjusted_204 = 2307.232878826027
 R2046: Sold_0[204] - Q0_var[204] - Q1_var[204]
   + total_stock_second_period_204 = 0
 R2047: total_stock_second_period_204 + Lost_1_expr_204
   = 1671.966694022811
 R2048: Sold_1[204] + Left_1[204] - total_stock_second_period_204 = 0
 R2049: - 700 Sold_0[204] - 700 Sold_1[204] + 700 Lost_0[204]
   + 700 Lost_1[204] + 300 Left_1[204] + profits_vars[204] = 0
 Fraction_lower_bound_205: Fraction[205] >= 0
 Fraction_upper_bound_205: Fraction[205] <= 1
 R2052: 1.504278000866664 alphas[0] + 1.312868553695165 alphas[1]
   + 0.4988237028700285 alphas[2] - 0.00251981300915956 alphas[3]
   - alphas[4] + f_var[205] = 0
 R2053: - 2156.17943612574 Fraction[205] + Q0_var[205] = 0
 R2054: Q0_var[205] + Lost_0_expr_205 = 484.0102663735216
 R2055: Q0_var[205] + Q_hat_adjusted_205 = 1615.084892616314
 R2056: Sold_0[205] - Q0_var[205] - Q1_var[205]
   + total_stock_second_period_205 = 0
 R2057: total_stock_second_period_205 + Lost_1_expr_205
   = 1027.965803012694
 R2058: Sold_1[205] + Left_1[205] - total_stock_second_period_205 = 0
 R2059: - 700 Sold_0[205] - 700 Sold_1[205] + 700 Lost_0[205]
   + 700 Lost_1[205] + 300 Left_1[205] + profits_vars[205] = 0
 Fraction_lower_bound_206: Fraction[206] >= 0
 Fraction_upper_bound_206: Fraction[206] <= 1
 R2062: - 0.6614087636151004 alphas[0] + 0.7504733567657628 alphas[1]
   - 0.1717005014136217 alphas[2] + 0.3318647906222814 alphas[3]
   - alphas[4] + f_var[206] = 0
 R2063: - 2156.17943612574 Fraction[206] + Q0_var[206] = 0
 R2064: Q0_var[206] + Lost_0_expr_206 = 187.2122906779878
 R2065: Q0_var[206] + Q_hat_adjusted_206 = 1849.897888382925
 R2066: Sold_0[206] - Q0_var[206] - Q1_var[206]
   + total_stock_second_period_206 = 0
 R2067: total_stock_second_period_206 + Lost_1_expr_206 = 1689.26904540914
 R2068: Sold_1[206] + Left_1[206] - total_stock_second_period_206 = 0
 R2069: - 700 Sold_0[206] - 700 Sold_1[206] + 700 Lost_0[206]
   + 700 Lost_1[206] + 300 Left_1[206] + profits_vars[206] = 0
 Fraction_lower_bound_207: Fraction[207] >= 0
 Fraction_upper_bound_207: Fraction[207] <= 1
 R2072: 0.1370755328569083 alphas[0] + 1.428276607914833 alphas[1]
   - 1.044849278159663 alphas[2] + 0.9526974199595536 alphas[3] - alphas[4]
   + f_var[207] = 0
 R2073: - 2156.17943612574 Fraction[207] + Q0_var[207] = 0
 R2074: Q0_var[207] + Lost_0_expr_207 = 385.6634294300526
 R2075: Q0_var[207] + Q_hat_adjusted_207 = 2530.972787909894
 R2076: Sold_0[207] - Q0_var[207] - Q1_var[207]
   + total_stock_second_period_207 = 0
 R2077: total_stock_second_period_207 + Lost_1_expr_207
   = 2063.274228670299
 R2078: Sold_1[207] + Left_1[207] - total_stock_second_period_207 = 0
 R2079: - 700 Sold_0[207] - 700 Sold_1[207] + 700 Lost_0[207]
   + 700 Lost_1[207] + 300 Left_1[207] + profits_vars[207] = 0
 Fraction_lower_bound_208: Fraction[208] >= 0
 Fraction_upper_bound_208: Fraction[208] <= 1
 R2082: - 0.1561520342367081 alphas[0] + 1.430132547178157 alphas[1]
   + 1.304130197179702 alphas[2] + 0.6951549880140478 alphas[3] - alphas[4]
   + f_var[208] = 0
 R2083: - 2156.17943612574 Fraction[208] + Q0_var[208] = 0
 R2084: Q0_var[208] + Lost_0_expr_208 = 175.2189204002573
 R2085: Q0_var[208] + Q_hat_adjusted_208 = 906.8623941414652
 R2086: Sold_0[208] - Q0_var[208] - Q1_var[208]
   + total_stock_second_period_208 = 0
 R2087: total_stock_second_period_208 + Lost_1_expr_208
   = 708.4006904256271
 R2088: Sold_1[208] + Left_1[208] - total_stock_second_period_208 = 0
 R2089: - 700 Sold_0[208] - 700 Sold_1[208] + 700 Lost_0[208]
   + 700 Lost_1[208] + 300 Left_1[208] + profits_vars[208] = 0
 Fraction_lower_bound_209: Fraction[209] >= 0
 Fraction_upper_bound_209: Fraction[209] <= 1
 R2092: - 1.430113306172345 alphas[0] - 1.264416950348514 alphas[1]
   - 0.998466699355714 alphas[2] - 1.447885729442971 alphas[3] - alphas[4]
   + f_var[209] = 0
 R2093: - 2156.17943612574 Fraction[209] + Q0_var[209] = 0
 R2094: Q0_var[209] + Lost_0_expr_209 = 402.0638126573399
 R2095: Q0_var[209] + Q_hat_adjusted_209 = 2512.079787452415
 R2096: Sold_0[209] - Q0_var[209] - Q1_var[209]
   + total_stock_second_period_209 = 0
 R2097: total_stock_second_period_209 + Lost_1_expr_209
   = 2108.421355066454
 R2098: Sold_1[209] + Left_1[209] - total_stock_second_period_209 = 0
 R2099: - 700 Sold_0[209] - 700 Sold_1[209] + 700 Lost_0[209]
   + 700 Lost_1[209] + 300 Left_1[209] + profits_vars[209] = 0
 Fraction_lower_bound_210: Fraction[210] >= 0
 Fraction_upper_bound_210: Fraction[210] <= 1
 R2102: - 1.761497888257432 alphas[0] + 1.101215206972098 alphas[1]
   + 1.492380944518889 alphas[2] + 0.741182448756684 alphas[3] - alphas[4]
   + f_var[210] = 0
 R2103: - 2156.17943612574 Fraction[210] + Q0_var[210] = 0
 R2104: Q0_var[210] + Lost_0_expr_210 = 130.2593793834359
 R2105: Q0_var[210] + Q_hat_adjusted_210 = 752.8177820466624
 R2106: Sold_0[210] - Q0_var[210] - Q1_var[210]
   + total_stock_second_period_210 = 0
 R2107: total_stock_second_period_210 + Lost_1_expr_210
   = 342.0926388199281
 R2108: Sold_1[210] + Left_1[210] - total_stock_second_period_210 = 0
 R2109: - 700 Sold_0[210] - 700 Sold_1[210] + 700 Lost_0[210]
   + 700 Lost_1[210] + 300 Left_1[210] + profits_vars[210] = 0
 Fraction_lower_bound_211: Fraction[211] >= 0
 Fraction_upper_bound_211: Fraction[211] <= 1
 R2112: 0.9748441653196667 alphas[0] - 0.2536472517215436 alphas[1]
   - 1.048769595789565 alphas[2] + 0.5404328060763408 alphas[3] - alphas[4]
   + f_var[211] = 0
 R2113: - 2156.17943612574 Fraction[211] + Q0_var[211] = 0
 R2114: Q0_var[211] + Lost_0_expr_211 = 331.8976489154021
 R2115: Q0_var[211] + Q_hat_adjusted_211 = 2498.962502164801
 R2116: Sold_0[211] - Q0_var[211] - Q1_var[211]
   + total_stock_second_period_211 = 0
 R2117: total_stock_second_period_211 + Lost_1_expr_211
   = 2165.498051142852
 R2118: Sold_1[211] + Left_1[211] - total_stock_second_period_211 = 0
 R2119: - 700 Sold_0[211] - 700 Sold_1[211] + 700 Lost_0[211]
   + 700 Lost_1[211] + 300 Left_1[211] + profits_vars[211] = 0
 Fraction_lower_bound_212: Fraction[212] >= 0
 Fraction_upper_bound_212: Fraction[212] <= 1
 R2122: - 0.6036898303233079 alphas[0] - 0.9821703361248986 alphas[1]
   - 1.019736636554276 alphas[2] + 1.717552455229473 alphas[3] - alphas[4]
   + f_var[212] = 0
 R2123: - 2156.17943612574 Fraction[212] + Q0_var[212] = 0
 R2124: Q0_var[212] + Lost_0_expr_212 = 368.9757114135849
 R2125: Q0_var[212] + Q_hat_adjusted_212 = 2504.339498250272
 R2126: Sold_0[212] - Q0_var[212] - Q1_var[212]
   + total_stock_second_period_212 = 0
 R2127: total_stock_second_period_212 + Lost_1_expr_212 = 2160.60213973027
 R2128: Sold_1[212] + Left_1[212] - total_stock_second_period_212 = 0
 R2129: - 700 Sold_0[212] - 700 Sold_1[212] + 700 Lost_0[212]
   + 700 Lost_1[212] + 300 Left_1[212] + profits_vars[212] = 0
 Fraction_lower_bound_213: Fraction[213] >= 0
 Fraction_upper_bound_213: Fraction[213] <= 1
 R2132: 0.8105968662493059 alphas[0] + 0.0982989692512982 alphas[1]
   + 1.460167486150379 alphas[2] - 1.110564762914233 alphas[3] - alphas[4]
   + f_var[213] = 0
 R2133: - 2156.17943612574 Fraction[213] + Q0_var[213] = 0
 R2134: Q0_var[213] + Lost_0_expr_213 = 63.00705450571935
 R2135: Q0_var[213] + Q_hat_adjusted_213 = 737.4062277261803
 R2136: Sold_0[213] - Q0_var[213] - Q1_var[213]
   + total_stock_second_period_213 = 0
 R2137: total_stock_second_period_213 + Lost_1_expr_213
   = 359.7222933776645
 R2138: Sold_1[213] + Left_1[213] - total_stock_second_period_213 = 0
 R2139: - 700 Sold_0[213] - 700 Sold_1[213] + 700 Lost_0[213]
   + 700 Lost_1[213] + 300 Left_1[213] + profits_vars[213] = 0
 Fraction_lower_bound_214: Fraction[214] >= 0
 Fraction_upper_bound_214: Fraction[214] <= 1
 R2142: 1.669138200434435 alphas[0] + 1.131039274052246 alphas[1]
   - 0.4283962989967022 alphas[2] + 1.209551130304257 alphas[3] - alphas[4]
   + f_var[214] = 0
 R2143: - 2156.17943612574 Fraction[214] + Q0_var[214] = 0
 R2144: Q0_var[214] + Lost_0_expr_214 = 425.7881502388163
 R2145: Q0_var[214] + Q_hat_adjusted_214 = 2165.765519374295
 R2146: Sold_0[214] - Q0_var[214] - Q1_var[214]
   + total_stock_second_period_214 = 0
 R2147: total_stock_second_period_214 + Lost_1_expr_214
   = 1782.476360684366
 R2148: Sold_1[214] + Left_1[214] - total_stock_second_period_214 = 0
 R2149: - 700 Sold_0[214] - 700 Sold_1[214] + 700 Lost_0[214]
   + 700 Lost_1[214] + 300 Left_1[214] + profits_vars[214] = 0
 Fraction_lower_bound_215: Fraction[215] >= 0
 Fraction_upper_bound_215: Fraction[215] <= 1
 R2152: - 0.940851044156333 alphas[0] + 0.9735001680667675 alphas[1]
   - 0.019619181349842 alphas[2] + 1.043825424690828 alphas[3] - alphas[4]
   + f_var[215] = 0
 R2153: - 2156.17943612574 Fraction[215] + Q0_var[215] = 0
 R2154: Q0_var[215] + Lost_0_expr_215 = 569.4957627222768
 R2155: Q0_var[215] + Q_hat_adjusted_215 = 1998.725765076419
 R2156: Sold_0[215] - Q0_var[215] - Q1_var[215]
   + total_stock_second_period_215 = 0
 R2157: total_stock_second_period_215 + Lost_1_expr_215
   = 1504.282122464128
 R2158: Sold_1[215] + Left_1[215] - total_stock_second_period_215 = 0
 R2159: - 700 Sold_0[215] - 700 Sold_1[215] + 700 Lost_0[215]
   + 700 Lost_1[215] + 300 Left_1[215] + profits_vars[215] = 0
 Fraction_lower_bound_216: Fraction[216] >= 0
 Fraction_upper_bound_216: Fraction[216] <= 1
 R2162: 0.6100177640419165 alphas[0] + 0.1781308646139686 alphas[1]
   + 1.07071494374674 alphas[2] + 0.8719932887266887 alphas[3] - alphas[4]
   + f_var[216] = 0
 R2163: - 2156.17943612574 Fraction[216] + Q0_var[216] = 0
 R2164: Q0_var[216] + Lost_0_expr_216 = 290.6983749604484
 R2165: Q0_var[216] + Q_hat_adjusted_216 = 1128.777363884514
 R2166: Sold_0[216] - Q0_var[216] - Q1_var[216]
   + total_stock_second_period_216 = 0
 R2167: total_stock_second_period_216 + Lost_1_expr_216
   = 832.9425905173721
 R2168: Sold_1[216] + Left_1[216] - total_stock_second_period_216 = 0
 R2169: - 700 Sold_0[216] - 700 Sold_1[216] + 700 Lost_0[216]
   + 700 Lost_1[216] + 300 Left_1[216] + profits_vars[216] = 0
 Fraction_lower_bound_217: Fraction[217] >= 0
 Fraction_upper_bound_217: Fraction[217] <= 1
 R2172: 0.3855510292444461 alphas[0] - 0.1464737088304863 alphas[1]
   + 1.140994621590573 alphas[2] - 1.24348674427005 alphas[3] - alphas[4]
   + f_var[217] = 0
 R2173: - 2156.17943612574 Fraction[217] + Q0_var[217] = 0
 R2174: Q0_var[217] + Lost_0_expr_217 = 203.3093280319208
 R2175: Q0_var[217] + Q_hat_adjusted_217 = 1027.129382598325
 R2176: Sold_0[217] - Q0_var[217] - Q1_var[217]
   + total_stock_second_period_217 = 0
 R2177: total_stock_second_period_217 + Lost_1_expr_217
   = 555.7965075185953
 R2178: Sold_1[217] + Left_1[217] - total_stock_second_period_217 = 0
 R2179: - 700 Sold_0[217] - 700 Sold_1[217] + 700 Lost_0[217]
   + 700 Lost_1[217] + 300 Left_1[217] + profits_vars[217] = 0
 Fraction_lower_bound_218: Fraction[218] >= 0
 Fraction_upper_bound_218: Fraction[218] <= 1
 R2182: - 0.3391943373398401 alphas[0] + 0.4628423605072734 alphas[1]
   - 0.9072898950940992 alphas[2] + 1.036070374075845 alphas[3] - alphas[4]
   + f_var[218] = 0
 R2183: - 2156.17943612574 Fraction[218] + Q0_var[218] = 0
 R2184: Q0_var[218] + Lost_0_expr_218 = 632.2731906317624
 R2185: Q0_var[218] + Q_hat_adjusted_218 = 2601.959973299749
 R2186: Sold_0[218] - Q0_var[218] - Q1_var[218]
   + total_stock_second_period_218 = 0
 R2187: total_stock_second_period_218 + Lost_1_expr_218
   = 2053.844029307897
 R2188: Sold_1[218] + Left_1[218] - total_stock_second_period_218 = 0
 R2189: - 700 Sold_0[218] - 700 Sold_1[218] + 700 Lost_0[218]
   + 700 Lost_1[218] + 300 Left_1[218] + profits_vars[218] = 0
 Fraction_lower_bound_219: Fraction[219] >= 0
 Fraction_upper_bound_219: Fraction[219] <= 1
 R2192: - 1.19794781819897 alphas[0] - 1.009515726233105 alphas[1]
   - 0.2334672400398104 alphas[2] - 0.9507291396317438 alphas[3]
   - alphas[4] + f_var[219] = 0
 R2193: - 2156.17943612574 Fraction[219] + Q0_var[219] = 0
 R2194: Q0_var[219] + Lost_0_expr_219 = 354.4425188202355
 R2195: Q0_var[219] + Q_hat_adjusted_219 = 1996.366177947571
 R2196: Sold_0[219] - Q0_var[219] - Q1_var[219]
   + total_stock_second_period_219 = 0
 R2197: total_stock_second_period_219 + Lost_1_expr_219
   = 1356.610419635541
 R2198: Sold_1[219] + Left_1[219] - total_stock_second_period_219 = 0
 R2199: - 700 Sold_0[219] - 700 Sold_1[219] + 700 Lost_0[219]
   + 700 Lost_1[219] + 300 Left_1[219] + profits_vars[219] = 0
 Fraction_lower_bound_220: Fraction[220] >= 0
 Fraction_upper_bound_220: Fraction[220] <= 1
 R2202: - 0.4830611253115499 alphas[0] - 0.9057172751023325 alphas[1]
   - 1.594793228753755 alphas[2] - 0.9861003738166172 alphas[3] - alphas[4]
   + f_var[220] = 0
 R2203: - 2156.17943612574 Fraction[220] + Q0_var[220] = 0
 R2204: Q0_var[220] + Lost_0_expr_220 = 595.6131967865671
 R2205: Q0_var[220] + Q_hat_adjusted_220 = 3014.450073855785
 R2206: Sold_0[220] - Q0_var[220] - Q1_var[220]
   + total_stock_second_period_220 = 0
 R2207: total_stock_second_period_220 + Lost_1_expr_220
   = 2152.986121828997
 R2208: Sold_1[220] + Left_1[220] - total_stock_second_period_220 = 0
 R2209: - 700 Sold_0[220] - 700 Sold_1[220] + 700 Lost_0[220]
   + 700 Lost_1[220] + 300 Left_1[220] + profits_vars[220] = 0
 Fraction_lower_bound_221: Fraction[221] >= 0
 Fraction_upper_bound_221: Fraction[221] <= 1
 R2212: - 1.345131259756711 alphas[0] - 1.506160336140665 alphas[1]
   - 1.592330365912356 alphas[2] - 1.386002245834323 alphas[3] - alphas[4]
   + f_var[221] = 0
 R2213: - 2156.17943612574 Fraction[221] + Q0_var[221] = 0
 R2214: Q0_var[221] + Lost_0_expr_221 = 712.1801777386459
 R2215: Q0_var[221] + Q_hat_adjusted_221 = 3087.678413736449
 R2216: Sold_0[221] - Q0_var[221] - Q1_var[221]
   + total_stock_second_period_221 = 0
 R2217: total_stock_second_period_221 + Lost_1_expr_221
   = 2297.202554404029
 R2218: Sold_1[221] + Left_1[221] - total_stock_second_period_221 = 0
 R2219: - 700 Sold_0[221] - 700 Sold_1[221] + 700 Lost_0[221]
   + 700 Lost_1[221] + 300 Left_1[221] + profits_vars[221] = 0
 Fraction_lower_bound_222: Fraction[222] >= 0
 Fraction_upper_bound_222: Fraction[222] <= 1
 R2222: 0.7744409793034601 alphas[0] + 1.555040943363305 alphas[1]
   - 1.626483343660187 alphas[2] + 1.341582224847912 alphas[3] - alphas[4]
   + f_var[222] = 0
 R2223: - 2156.17943612574 Fraction[222] + Q0_var[222] = 0
 R2224: Q0_var[222] + Lost_0_expr_222 = 406.7527289161615
 R2225: Q0_var[222] + Q_hat_adjusted_222 = 2913.373478447863
 R2226: Sold_0[222] - Q0_var[222] - Q1_var[222]
   + total_stock_second_period_222 = 0
 R2227: total_stock_second_period_222 + Lost_1_expr_222 = 2516.69482811062
 R2228: Sold_1[222] + Left_1[222] - total_stock_second_period_222 = 0
 R2229: - 700 Sold_0[222] - 700 Sold_1[222] + 700 Lost_0[222]
   + 700 Lost_1[222] + 300 Left_1[222] + profits_vars[222] = 0
 Fraction_lower_bound_223: Fraction[223] >= 0
 Fraction_upper_bound_223: Fraction[223] <= 1
 R2232: - 1.081192161557318 alphas[0] - 1.398740742619085 alphas[1]
   + 1.238555645403269 alphas[2] + 0.9996753396981647 alphas[3] - alphas[4]
   + f_var[223] = 0
 R2233: - 2156.17943612574 Fraction[223] + Q0_var[223] = 0
 R2234: Q0_var[223] + Lost_0_expr_223 = 157.1787664812613
 R2235: Q0_var[223] + Q_hat_adjusted_223 = 930.4458100972213
 R2236: Sold_0[223] - Q0_var[223] - Q1_var[223]
   + total_stock_second_period_223 = 0
 R2237: total_stock_second_period_223 + Lost_1_expr_223
   = 1110.656597514951
 R2238: Sold_1[223] + Left_1[223] - total_stock_second_period_223 = 0
 R2239: - 700 Sold_0[223] - 700 Sold_1[223] + 700 Lost_0[223]
   + 700 Lost_1[223] + 300 Left_1[223] + profits_vars[223] = 0
 Fraction_lower_bound_224: Fraction[224] >= 0
 Fraction_upper_bound_224: Fraction[224] <= 1
 R2242: 1.085441860733205 alphas[0] + 0.3158636267199662 alphas[1]
   - 1.448130889894074 alphas[2] - 0.7671060296171698 alphas[3] - alphas[4]
   + f_var[224] = 0
 R2243: - 2156.17943612574 Fraction[224] + Q0_var[224] = 0
 R2244: Q0_var[224] + Lost_0_expr_224 = 586.600381579033
 R2245: Q0_var[224] + Q_hat_adjusted_224 = 2915.654811331396
 R2246: Sold_0[224] - Q0_var[224] - Q1_var[224]
   + total_stock_second_period_224 = 0
 R2247: total_stock_second_period_224 + Lost_1_expr_224
   = 1996.744526077786
 R2248: Sold_1[224] + Left_1[224] - total_stock_second_period_224 = 0
 R2249: - 700 Sold_0[224] - 700 Sold_1[224] + 700 Lost_0[224]
   + 700 Lost_1[224] + 300 Left_1[224] + profits_vars[224] = 0
 Fraction_lower_bound_225: Fraction[225] >= 0
 Fraction_upper_bound_225: Fraction[225] <= 1
 R2252: - 1.628660216263447 alphas[0] - 1.339719564198374 alphas[1]
   + 0.8256927982610462 alphas[2] + 1.071198508317403 alphas[3] - alphas[4]
   + f_var[225] = 0
 R2253: - 2156.17943612574 Fraction[225] + Q0_var[225] = 0
 R2254: Q0_var[225] + Lost_0_expr_225 = 175.8431246798929
 R2255: Q0_var[225] + Q_hat_adjusted_225 = 1210.192223221027
 R2256: Sold_0[225] - Q0_var[225] - Q1_var[225]
   + total_stock_second_period_225 = 0
 R2257: total_stock_second_period_225 + Lost_1_expr_225
   = 1009.310342243003
 R2258: Sold_1[225] + Left_1[225] - total_stock_second_period_225 = 0
 R2259: - 700 Sold_0[225] - 700 Sold_1[225] + 700 Lost_0[225]
   + 700 Lost_1[225] + 300 Left_1[225] + profits_vars[225] = 0
 Fraction_lower_bound_226: Fraction[226] >= 0
 Fraction_upper_bound_226: Fraction[226] <= 1
 R2262: - 0.690049652085687 alphas[0] - 0.7004344571755108 alphas[1]
   + 0.1253970587882675 alphas[2] + 0.7675733905351756 alphas[3]
   - alphas[4] + f_var[226] = 0
 R2263: - 2156.17943612574 Fraction[226] + Q0_var[226] = 0
 R2264: Q0_var[226] + Lost_0_expr_226 = 324.8468549800301
 R2265: Q0_var[226] + Q_hat_adjusted_226 = 1749.786329470956
 R2266: Sold_0[226] - Q0_var[226] - Q1_var[226]
   + total_stock_second_period_226 = 0
 R2267: total_stock_second_period_226 + Lost_1_expr_226
   = 1703.568464256265
 R2268: Sold_1[226] + Left_1[226] - total_stock_second_period_226 = 0
 R2269: - 700 Sold_0[226] - 700 Sold_1[226] + 700 Lost_0[226]
   + 700 Lost_1[226] + 300 Left_1[226] + profits_vars[226] = 0
 Fraction_lower_bound_227: Fraction[227] >= 0
 Fraction_upper_bound_227: Fraction[227] <= 1
 R2272: 0.9797593627500195 alphas[0] - 1.71110262385908 alphas[1]
   - 0.8553270151910298 alphas[2] + 1.169548325118835 alphas[3] - alphas[4]
   + f_var[227] = 0
 R2273: - 2156.17943612574 Fraction[227] + Q0_var[227] = 0
 R2274: Q0_var[227] + Lost_0_expr_227 = 484.0649891446412
 R2275: Q0_var[227] + Q_hat_adjusted_227 = 2473.913821349764
 R2276: Sold_0[227] - Q0_var[227] - Q1_var[227]
   + total_stock_second_period_227 = 0
 R2277: total_stock_second_period_227 + Lost_1_expr_227 = 1862.01160449829
 R2278: Sold_1[227] + Left_1[227] - total_stock_second_period_227 = 0
 R2279: - 700 Sold_0[227] - 700 Sold_1[227] + 700 Lost_0[227]
   + 700 Lost_1[227] + 300 Left_1[227] + profits_vars[227] = 0
 Fraction_lower_bound_228: Fraction[228] >= 0
 Fraction_upper_bound_228: Fraction[228] <= 1
 R2282: - 1.609481153189054 alphas[0] - 0.9338694817738751 alphas[1]
   + 0.9599177945405677 alphas[2] + 1.149531232871485 alphas[3] - alphas[4]
   + f_var[228] = 0
 R2283: - 2156.17943612574 Fraction[228] + Q0_var[228] = 0
 R2284: Q0_var[228] + Lost_0_expr_228 = 347.8365761996418
 R2285: Q0_var[228] + Q_hat_adjusted_228 = 1236.557754829561
 R2286: Sold_0[228] - Q0_var[228] - Q1_var[228]
   + total_stock_second_period_228 = 0
 R2287: total_stock_second_period_228 + Lost_1_expr_228
   = 855.3450909852002
 R2288: Sold_1[228] + Left_1[228] - total_stock_second_period_228 = 0
 R2289: - 700 Sold_0[228] - 700 Sold_1[228] + 700 Lost_0[228]
   + 700 Lost_1[228] + 300 Left_1[228] + profits_vars[228] = 0
 Fraction_lower_bound_229: Fraction[229] >= 0
 Fraction_upper_bound_229: Fraction[229] <= 1
 R2292: - 0.8434786135619582 alphas[0] + 0.4113679047988248 alphas[1]
   - 0.2289400131354223 alphas[2] + 0.1113612357829089 alphas[3]
   - alphas[4] + f_var[229] = 0
 R2293: - 2156.17943612574 Fraction[229] + Q0_var[229] = 0
 R2294: Q0_var[229] + Lost_0_expr_229 = 471.2941132357026
 R2295: Q0_var[229] + Q_hat_adjusted_229 = 2068.468127261324
 R2296: Sold_0[229] - Q0_var[229] - Q1_var[229]
   + total_stock_second_period_229 = 0
 R2297: total_stock_second_period_229 + Lost_1_expr_229
   = 1268.016638120852
 R2298: Sold_1[229] + Left_1[229] - total_stock_second_period_229 = 0
 R2299: - 700 Sold_0[229] - 700 Sold_1[229] + 700 Lost_0[229]
   + 700 Lost_1[229] + 300 Left_1[229] + profits_vars[229] = 0
 Fraction_lower_bound_230: Fraction[230] >= 0
 Fraction_upper_bound_230: Fraction[230] <= 1
 R2302: 0.8437847440145857 alphas[0] - 0.0538772305820728 alphas[1]
   + 1.018277836661082 alphas[2] + 1.361340983367462 alphas[3] - alphas[4]
   + f_var[230] = 0
 R2303: - 2156.17943612574 Fraction[230] + Q0_var[230] = 0
 R2304: Q0_var[230] + Lost_0_expr_230 = 132.9830372801353
 R2305: Q0_var[230] + Q_hat_adjusted_230 = 1060.472543093716
 R2306: Sold_0[230] - Q0_var[230] - Q1_var[230]
   + total_stock_second_period_230 = 0
 R2307: total_stock_second_period_230 + Lost_1_expr_230 = 614.45685255019
 R2308: Sold_1[230] + Left_1[230] - total_stock_second_period_230 = 0
 R2309: - 700 Sold_0[230] - 700 Sold_1[230] + 700 Lost_0[230]
   + 700 Lost_1[230] + 300 Left_1[230] + profits_vars[230] = 0
 Fraction_lower_bound_231: Fraction[231] >= 0
 Fraction_upper_bound_231: Fraction[231] <= 1
 R2312: 0.9875274770231659 alphas[0] + 0.3710026231746032 alphas[1]
   + 1.284050741234776 alphas[2] + 0.6486895375126209 alphas[3] - alphas[4]
   + f_var[231] = 0
 R2313: - 2156.17943612574 Fraction[231] + Q0_var[231] = 0
 R2314: Q0_var[231] + Lost_0_expr_231 = 101.1916516040443
 R2315: Q0_var[231] + Q_hat_adjusted_231 = 870.3480277687609
 R2316: Sold_0[231] - Q0_var[231] - Q1_var[231]
   + total_stock_second_period_231 = 0
 R2317: total_stock_second_period_231 + Lost_1_expr_231
   = 803.9628061026388
 R2318: Sold_1[231] + Left_1[231] - total_stock_second_period_231 = 0
 R2319: - 700 Sold_0[231] - 700 Sold_1[231] + 700 Lost_0[231]
   + 700 Lost_1[231] + 300 Left_1[231] + profits_vars[231] = 0
 Fraction_lower_bound_232: Fraction[232] >= 0
 Fraction_upper_bound_232: Fraction[232] <= 1
 R2322: - 0.0911311831376505 alphas[0] + 0.4101164801279929 alphas[1]
   + 1.245297943622365 alphas[2] + 0.2960982115721975 alphas[3] - alphas[4]
   + f_var[232] = 0
 R2323: - 2156.17943612574 Fraction[232] + Q0_var[232] = 0
 R2324: Q0_var[232] + Lost_0_expr_232 = 268.7082871069592
 R2325: Q0_var[232] + Q_hat_adjusted_232 = 1002.864514780808
 R2326: Sold_0[232] - Q0_var[232] - Q1_var[232]
   + total_stock_second_period_232 = 0
 R2327: total_stock_second_period_232 + Lost_1_expr_232
   = 897.8934453526914
 R2328: Sold_1[232] + Left_1[232] - total_stock_second_period_232 = 0
 R2329: - 700 Sold_0[232] - 700 Sold_1[232] + 700 Lost_0[232]
   + 700 Lost_1[232] + 300 Left_1[232] + profits_vars[232] = 0
 Fraction_lower_bound_233: Fraction[233] >= 0
 Fraction_upper_bound_233: Fraction[233] <= 1
 R2332: 1.651407359606929 alphas[0] + 0.764565809809178 alphas[1]
   + 0.4550643825481779 alphas[2] + 1.647507853751646 alphas[3] - alphas[4]
   + f_var[233] = 0
 R2333: - 2156.17943612574 Fraction[233] + Q0_var[233] = 0
 R2334: Q0_var[233] + Lost_0_expr_233 = 277.0384014393732
 R2335: Q0_var[233] + Q_hat_adjusted_233 = 1510.045258935486
 R2336: Sold_0[233] - Q0_var[233] - Q1_var[233]
   + total_stock_second_period_233 = 0
 R2337: total_stock_second_period_233 + Lost_1_expr_233
   = 944.7777870953657
 R2338: Sold_1[233] + Left_1[233] - total_stock_second_period_233 = 0
 R2339: - 700 Sold_0[233] - 700 Sold_1[233] + 700 Lost_0[233]
   + 700 Lost_1[233] + 300 Left_1[233] + profits_vars[233] = 0
 Fraction_lower_bound_234: Fraction[234] >= 0
 Fraction_upper_bound_234: Fraction[234] <= 1
 R2342: 1.008195403873216 alphas[0] - 0.0365210799429588 alphas[1]
   + 1.61210549355476 alphas[2] + 0.4247052958046842 alphas[3] - alphas[4]
   + f_var[234] = 0
 R2343: - 2156.17943612574 Fraction[234] + Q0_var[234] = 0
 R2344: Q0_var[234] + Lost_0_expr_234 = 160.0402907632731
 R2345: Q0_var[234] + Q_hat_adjusted_234 = 695.5200199914852
 R2346: Sold_0[234] - Q0_var[234] - Q1_var[234]
   + total_stock_second_period_234 = 0
 R2347: total_stock_second_period_234 + Lost_1_expr_234
   = 679.1398069850899
 R2348: Sold_1[234] + Left_1[234] - total_stock_second_period_234 = 0
 R2349: - 700 Sold_0[234] - 700 Sold_1[234] + 700 Lost_0[234]
   + 700 Lost_1[234] + 300 Left_1[234] + profits_vars[234] = 0
 Fraction_lower_bound_235: Fraction[235] >= 0
 Fraction_upper_bound_235: Fraction[235] <= 1
 R2352: 0.2397141941190356 alphas[0] - 0.6696103401517514 alphas[1]
   + 0.234741809410868 alphas[2] + 1.413645423344317 alphas[3] - alphas[4]
   + f_var[235] = 0
 R2353: - 2156.17943612574 Fraction[235] + Q0_var[235] = 0
 R2354: Q0_var[235] + Lost_0_expr_235 = 156.2134333933114
 R2355: Q0_var[235] + Q_hat_adjusted_235 = 1572.244777436297
 R2356: Sold_0[235] - Q0_var[235] - Q1_var[235]
   + total_stock_second_period_235 = 0
 R2357: total_stock_second_period_235 + Lost_1_expr_235 = 1285.29531811116
 R2358: Sold_1[235] + Left_1[235] - total_stock_second_period_235 = 0
 R2359: - 700 Sold_0[235] - 700 Sold_1[235] + 700 Lost_0[235]
   + 700 Lost_1[235] + 300 Left_1[235] + profits_vars[235] = 0
 Fraction_lower_bound_236: Fraction[236] >= 0
 Fraction_upper_bound_236: Fraction[236] <= 1
 R2362: 0.4184317475051419 alphas[0] + 0.708553911711315 alphas[1]
   - 0.5497972859598683 alphas[2] + 0.0709406175965907 alphas[3]
   - alphas[4] + f_var[236] = 0
 R2363: - 2156.17943612574 Fraction[236] + Q0_var[236] = 0
 R2364: Q0_var[236] + Lost_0_expr_236 = 558.9120703340504
 R2365: Q0_var[236] + Q_hat_adjusted_236 = 2328.17080809399
 R2366: Sold_0[236] - Q0_var[236] - Q1_var[236]
   + total_stock_second_period_236 = 0
 R2367: total_stock_second_period_236 + Lost_1_expr_236
   = 1624.548751157519
 R2368: Sold_1[236] + Left_1[236] - total_stock_second_period_236 = 0
 R2369: - 700 Sold_0[236] - 700 Sold_1[236] + 700 Lost_0[236]
   + 700 Lost_1[236] + 300 Left_1[236] + profits_vars[236] = 0
 Fraction_lower_bound_237: Fraction[237] >= 0
 Fraction_upper_bound_237: Fraction[237] <= 1
 R2372: 0.1021263386658449 alphas[0] - 0.1333386134271443 alphas[1]
   + 0.3106763959971924 alphas[2] - 1.045558476536479 alphas[3] - alphas[4]
   + f_var[237] = 0
 R2373: - 2156.17943612574 Fraction[237] + Q0_var[237] = 0
 R2374: Q0_var[237] + Lost_0_expr_237 = 298.9692339494929
 R2375: Q0_var[237] + Q_hat_adjusted_237 = 1615.680146257588
 R2376: Sold_0[237] - Q0_var[237] - Q1_var[237]
   + total_stock_second_period_237 = 0
 R2377: total_stock_second_period_237 + Lost_1_expr_237
   = 1473.714354490466
 R2378: Sold_1[237] + Left_1[237] - total_stock_second_period_237 = 0
 R2379: - 700 Sold_0[237] - 700 Sold_1[237] + 700 Lost_0[237]
   + 700 Lost_1[237] + 300 Left_1[237] + profits_vars[237] = 0
 Fraction_lower_bound_238: Fraction[238] >= 0
 Fraction_upper_bound_238: Fraction[238] <= 1
 R2382: 0.7599828407974289 alphas[0] + 1.253685712051107 alphas[1]
   + 1.346138793495139 alphas[2] - 1.020966878374451 alphas[3] - alphas[4]
   + f_var[238] = 0
 R2383: - 2156.17943612574 Fraction[238] + Q0_var[238] = 0
 R2384: Q0_var[238] + Lost_0_expr_238 = 272.8704498859149
 R2385: Q0_var[238] + Q_hat_adjusted_238 = 942.3416185223085
 R2386: Sold_0[238] - Q0_var[238] - Q1_var[238]
   + total_stock_second_period_238 = 0
 R2387: total_stock_second_period_238 + Lost_1_expr_238
   = 593.3703776561499
 R2388: Sold_1[238] + Left_1[238] - total_stock_second_period_238 = 0
 R2389: - 700 Sold_0[238] - 700 Sold_1[238] + 700 Lost_0[238]
   + 700 Lost_1[238] + 300 Left_1[238] + profits_vars[238] = 0
 Fraction_lower_bound_239: Fraction[239] >= 0
 Fraction_upper_bound_239: Fraction[239] <= 1
 R2392: - 0.333771596834109 alphas[0] + 1.108968150022738 alphas[1]
   + 1.42187671460012 alphas[2] + 1.459608018634246 alphas[3] - alphas[4]
   + f_var[239] = 0
 R2393: - 2156.17943612574 Fraction[239] + Q0_var[239] = 0
 R2394: Q0_var[239] + Lost_0_expr_239 = 65.62455546577618
 R2395: Q0_var[239] + Q_hat_adjusted_239 = 760.8508753075872
 R2396: Sold_0[239] - Q0_var[239] - Q1_var[239]
   + total_stock_second_period_239 = 0
 R2397: total_stock_second_period_239 + Lost_1_expr_239
   = 435.0255291043418
 R2398: Sold_1[239] + Left_1[239] - total_stock_second_period_239 = 0
 R2399: - 700 Sold_0[239] - 700 Sold_1[239] + 700 Lost_0[239]
   + 700 Lost_1[239] + 300 Left_1[239] + profits_vars[239] = 0
 Fraction_lower_bound_240: Fraction[240] >= 0
 Fraction_upper_bound_240: Fraction[240] <= 1
 R2402: - 1.314015461768494 alphas[0] + 1.494196490829341 alphas[1]
   + 0.7157299529046628 alphas[2] - 0.2909042399031625 alphas[3]
   - alphas[4] + f_var[240] = 0
 R2403: - 2156.17943612574 Fraction[240] + Q0_var[240] = 0
 R2404: Q0_var[240] + Lost_0_expr_240 = 242.5892901516935
 R2405: Q0_var[240] + Q_hat_adjusted_240 = 1322.612318936668
 R2406: Sold_0[240] - Q0_var[240] - Q1_var[240]
   + total_stock_second_period_240 = 0
 R2407: total_stock_second_period_240 + Lost_1_expr_240
   = 1000.828354995933
 R2408: Sold_1[240] + Left_1[240] - total_stock_second_period_240 = 0
 R2409: - 700 Sold_0[240] - 700 Sold_1[240] + 700 Lost_0[240]
   + 700 Lost_1[240] + 300 Left_1[240] + profits_vars[240] = 0
 Fraction_lower_bound_241: Fraction[241] >= 0
 Fraction_upper_bound_241: Fraction[241] <= 1
 R2412: 1.326385706388528 alphas[0] - 1.654472042727139 alphas[1]
   + 1.072376216544812 alphas[2] - 1.075300197757549 alphas[3] - alphas[4]
   + f_var[241] = 0
 R2413: - 2156.17943612574 Fraction[241] + Q0_var[241] = 0
 R2414: Q0_var[241] + Lost_0_expr_241 = 180.6812511953311
 R2415: Q0_var[241] + Q_hat_adjusted_241 = 1056.17441365287
 R2416: Sold_0[241] - Q0_var[241] - Q1_var[241]
   + total_stock_second_period_241 = 0
 R2417: total_stock_second_period_241 + Lost_1_expr_241
   = 612.2469752146003
 R2418: Sold_1[241] + Left_1[241] - total_stock_second_period_241 = 0
 R2419: - 700 Sold_0[241] - 700 Sold_1[241] + 700 Lost_0[241]
   + 700 Lost_1[241] + 300 Left_1[241] + profits_vars[241] = 0
 Fraction_lower_bound_242: Fraction[242] >= 0
 Fraction_upper_bound_242: Fraction[242] <= 1
 R2422: - 0.0882244399277781 alphas[0] + 1.640553301215436 alphas[1]
   - 1.536105470887893 alphas[2] + 0.9301173656630229 alphas[3] - alphas[4]
   + f_var[242] = 0
 R2423: - 2156.17943612574 Fraction[242] + Q0_var[242] = 0
 R2424: Q0_var[242] + Lost_0_expr_242 = 587.8996746666605
 R2425: Q0_var[242] + Q_hat_adjusted_242 = 2972.281522566339
 R2426: Sold_0[242] - Q0_var[242] - Q1_var[242]
   + total_stock_second_period_242 = 0
 R2427: total_stock_second_period_242 + Lost_1_expr_242
   = 2523.588319787366
 R2428: Sold_1[242] + Left_1[242] - total_stock_second_period_242 = 0
 R2429: - 700 Sold_0[242] - 700 Sold_1[242] + 700 Lost_0[242]
   + 700 Lost_1[242] + 300 Left_1[242] + profits_vars[242] = 0
 Fraction_lower_bound_243: Fraction[243] >= 0
 Fraction_upper_bound_243: Fraction[243] <= 1
 R2432: 1.274958249478294 alphas[0] + 1.045121738713003 alphas[1]
   + 1.131648433743716 alphas[2] - 0.8310662042329419 alphas[3] - alphas[4]
   + f_var[243] = 0
 R2433: - 2156.17943612574 Fraction[243] + Q0_var[243] = 0
 R2434: Q0_var[243] + Lost_0_expr_243 = 217.72846961345
 R2435: Q0_var[243] + Q_hat_adjusted_243 = 1043.117005941542
 R2436: Sold_0[243] - Q0_var[243] - Q1_var[243]
   + total_stock_second_period_243 = 0
 R2437: total_stock_second_period_243 + Lost_1_expr_243
   = 974.3888331272441
 R2438: Sold_1[243] + Left_1[243] - total_stock_second_period_243 = 0
 R2439: - 700 Sold_0[243] - 700 Sold_1[243] + 700 Lost_0[243]
   + 700 Lost_1[243] + 300 Left_1[243] + profits_vars[243] = 0
 Fraction_lower_bound_244: Fraction[244] >= 0
 Fraction_upper_bound_244: Fraction[244] <= 1
 R2442: - 0.7939620514445204 alphas[0] - 0.4348955833940029 alphas[1]
   - 1.01284860094703 alphas[2] - 1.264133022730294 alphas[3] - alphas[4]
   + f_var[244] = 0
 R2443: - 2156.17943612574 Fraction[244] + Q0_var[244] = 0
 R2444: Q0_var[244] + Lost_0_expr_244 = 403.1149483496066
 R2445: Q0_var[244] + Q_hat_adjusted_244 = 2521.875140026048
 R2446: Sold_0[244] - Q0_var[244] - Q1_var[244]
   + total_stock_second_period_244 = 0
 R2447: total_stock_second_period_244 + Lost_1_expr_244
   = 2039.707295387194
 R2448: Sold_1[244] + Left_1[244] - total_stock_second_period_244 = 0
 R2449: - 700 Sold_0[244] - 700 Sold_1[244] + 700 Lost_0[244]
   + 700 Lost_1[244] + 300 Left_1[244] + profits_vars[244] = 0
 Fraction_lower_bound_245: Fraction[245] >= 0
 Fraction_upper_bound_245: Fraction[245] <= 1
 R2452: 0.340988615363727 alphas[0] + 1.376407504848391 alphas[1]
   - 0.2938300987101199 alphas[2] - 0.9796291736111 alphas[3] - alphas[4]
   + f_var[245] = 0
 R2453: - 2156.17943612574 Fraction[245] + Q0_var[245] = 0
 R2454: Q0_var[245] + Lost_0_expr_245 = 209.0748380809555
 R2455: Q0_var[245] + Q_hat_adjusted_245 = 1941.379284672831
 R2456: Sold_0[245] - Q0_var[245] - Q1_var[245]
   + total_stock_second_period_245 = 0
 R2457: total_stock_second_period_245 + Lost_1_expr_245
   = 1516.009586511807
 R2458: Sold_1[245] + Left_1[245] - total_stock_second_period_245 = 0
 R2459: - 700 Sold_0[245] - 700 Sold_1[245] + 700 Lost_0[245]
   + 700 Lost_1[245] + 300 Left_1[245] + profits_vars[245] = 0
 Fraction_lower_bound_246: Fraction[246] >= 0
 Fraction_upper_bound_246: Fraction[246] <= 1
 R2462: - 0.2581918212805997 alphas[0] - 1.351729122498534 alphas[1]
   - 0.2386927474873786 alphas[2] + 1.369347780535381 alphas[3] - alphas[4]
   + f_var[246] = 0
 R2463: - 2156.17943612574 Fraction[246] + Q0_var[246] = 0
 R2464: Q0_var[246] + Lost_0_expr_246 = 448.1706425341932
 R2465: Q0_var[246] + Q_hat_adjusted_246 = 2059.818765080974
 R2466: Sold_0[246] - Q0_var[246] - Q1_var[246]
   + total_stock_second_period_246 = 0
 R2467: total_stock_second_period_246 + Lost_1_expr_246
   = 1292.533039627259
 R2468: Sold_1[246] + Left_1[246] - total_stock_second_period_246 = 0
 R2469: - 700 Sold_0[246] - 700 Sold_1[246] + 700 Lost_0[246]
   + 700 Lost_1[246] + 300 Left_1[246] + profits_vars[246] = 0
 Fraction_lower_bound_247: Fraction[247] >= 0
 Fraction_upper_bound_247: Fraction[247] <= 1
 R2472: 1.09377747990439 alphas[0] - 0.7987033105062857 alphas[1]
   - 0.0511774698188519 alphas[2] + 1.699568566774096 alphas[3] - alphas[4]
   + f_var[247] = 0
 R2473: - 2156.17943612574 Fraction[247] + Q0_var[247] = 0
 R2474: Q0_var[247] + Lost_0_expr_247 = 523.2013456654935
 R2475: Q0_var[247] + Q_hat_adjusted_247 = 1989.038125474763
 R2476: Sold_0[247] - Q0_var[247] - Q1_var[247]
   + total_stock_second_period_247 = 0
 R2477: total_stock_second_period_247 + Lost_1_expr_247
   = 1671.612124024781
 R2478: Sold_1[247] + Left_1[247] - total_stock_second_period_247 = 0
 R2479: - 700 Sold_0[247] - 700 Sold_1[247] + 700 Lost_0[247]
   + 700 Lost_1[247] + 300 Left_1[247] + profits_vars[247] = 0
 Fraction_lower_bound_248: Fraction[248] >= 0
 Fraction_upper_bound_248: Fraction[248] <= 1
 R2482: 1.2297454169684 alphas[0] - 1.63967662873588 alphas[1]
   - 1.557962735962494 alphas[2] + 1.085573732538948 alphas[3] - alphas[4]
   + f_var[248] = 0
 R2483: - 2156.17943612574 Fraction[248] + Q0_var[248] = 0
 R2484: Q0_var[248] + Lost_0_expr_248 = 533.2026437640332
 R2485: Q0_var[248] + Q_hat_adjusted_248 = 2951.049264996444
 R2486: Sold_0[248] - Q0_var[248] - Q1_var[248]
   + total_stock_second_period_248 = 0
 R2487: total_stock_second_period_248 + Lost_1_expr_248
   = 2283.740604754639
 R2488: Sold_1[248] + Left_1[248] - total_stock_second_period_248 = 0
 R2489: - 700 Sold_0[248] - 700 Sold_1[248] + 700 Lost_0[248]
   + 700 Lost_1[248] + 300 Left_1[248] + profits_vars[248] = 0
 Fraction_lower_bound_249: Fraction[249] >= 0
 Fraction_upper_bound_249: Fraction[249] <= 1
 R2492: 0.0155160954121656 alphas[0] - 0.0762753976525781 alphas[1]
   + 0.4347226884261101 alphas[2] + 0.8208579275414748 alphas[3]
   - alphas[4] + f_var[249] = 0
 R2493: - 2156.17943612574 Fraction[249] + Q0_var[249] = 0
 R2494: Q0_var[249] + Lost_0_expr_249 = 199.9707338281307
 R2495: Q0_var[249] + Q_hat_adjusted_249 = 1473.476594135979
 R2496: Sold_0[249] - Q0_var[249] - Q1_var[249]
   + total_stock_second_period_249 = 0
 R2497: total_stock_second_period_249 + Lost_1_expr_249
   = 916.6298460252046
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
 Q0_var[0] <= 2157.17943612574
 Q0_var[1] <= 2157.17943612574
 Q0_var[2] <= 2157.17943612574
 Q0_var[3] <= 2157.17943612574
 Q0_var[4] <= 2157.17943612574
 Q0_var[5] <= 2157.17943612574
 Q0_var[6] <= 2157.17943612574
 Q0_var[7] <= 2157.17943612574
 Q0_var[8] <= 2157.17943612574
 Q0_var[9] <= 2157.17943612574
 Q0_var[10] <= 2157.17943612574
 Q0_var[11] <= 2157.17943612574
 Q0_var[12] <= 2157.17943612574
 Q0_var[13] <= 2157.17943612574
 Q0_var[14] <= 2157.17943612574
 Q0_var[15] <= 2157.17943612574
 Q0_var[16] <= 2157.17943612574
 Q0_var[17] <= 2157.17943612574
 Q0_var[18] <= 2157.17943612574
 Q0_var[19] <= 2157.17943612574
 Q0_var[20] <= 2157.17943612574
 Q0_var[21] <= 2157.17943612574
 Q0_var[22] <= 2157.17943612574
 Q0_var[23] <= 2157.17943612574
 Q0_var[24] <= 2157.17943612574
 Q0_var[25] <= 2157.17943612574
 Q0_var[26] <= 2157.17943612574
 Q0_var[27] <= 2157.17943612574
 Q0_var[28] <= 2157.17943612574
 Q0_var[29] <= 2157.17943612574
 Q0_var[30] <= 2157.17943612574
 Q0_var[31] <= 2157.17943612574
 Q0_var[32] <= 2157.17943612574
 Q0_var[33] <= 2157.17943612574
 Q0_var[34] <= 2157.17943612574
 Q0_var[35] <= 2157.17943612574
 Q0_var[36] <= 2157.17943612574
 Q0_var[37] <= 2157.17943612574
 Q0_var[38] <= 2157.17943612574
 Q0_var[39] <= 2157.17943612574
 Q0_var[40] <= 2157.17943612574
 Q0_var[41] <= 2157.17943612574
 Q0_var[42] <= 2157.17943612574
 Q0_var[43] <= 2157.17943612574
 Q0_var[44] <= 2157.17943612574
 Q0_var[45] <= 2157.17943612574
 Q0_var[46] <= 2157.17943612574
 Q0_var[47] <= 2157.17943612574
 Q0_var[48] <= 2157.17943612574
 Q0_var[49] <= 2157.17943612574
 Q0_var[50] <= 2157.17943612574
 Q0_var[51] <= 2157.17943612574
 Q0_var[52] <= 2157.17943612574
 Q0_var[53] <= 2157.17943612574
 Q0_var[54] <= 2157.17943612574
 Q0_var[55] <= 2157.17943612574
 Q0_var[56] <= 2157.17943612574
 Q0_var[57] <= 2157.17943612574
 Q0_var[58] <= 2157.17943612574
 Q0_var[59] <= 2157.17943612574
 Q0_var[60] <= 2157.17943612574
 Q0_var[61] <= 2157.17943612574
 Q0_var[62] <= 2157.17943612574
 Q0_var[63] <= 2157.17943612574
 Q0_var[64] <= 2157.17943612574
 Q0_var[65] <= 2157.17943612574
 Q0_var[66] <= 2157.17943612574
 Q0_var[67] <= 2157.17943612574
 Q0_var[68] <= 2157.17943612574
 Q0_var[69] <= 2157.17943612574
 Q0_var[70] <= 2157.17943612574
 Q0_var[71] <= 2157.17943612574
 Q0_var[72] <= 2157.17943612574
 Q0_var[73] <= 2157.17943612574
 Q0_var[74] <= 2157.17943612574
 Q0_var[75] <= 2157.17943612574
 Q0_var[76] <= 2157.17943612574
 Q0_var[77] <= 2157.17943612574
 Q0_var[78] <= 2157.17943612574
 Q0_var[79] <= 2157.17943612574
 Q0_var[80] <= 2157.17943612574
 Q0_var[81] <= 2157.17943612574
 Q0_var[82] <= 2157.17943612574
 Q0_var[83] <= 2157.17943612574
 Q0_var[84] <= 2157.17943612574
 Q0_var[85] <= 2157.17943612574
 Q0_var[86] <= 2157.17943612574
 Q0_var[87] <= 2157.17943612574
 Q0_var[88] <= 2157.17943612574
 Q0_var[89] <= 2157.17943612574
 Q0_var[90] <= 2157.17943612574
 Q0_var[91] <= 2157.17943612574
 Q0_var[92] <= 2157.17943612574
 Q0_var[93] <= 2157.17943612574
 Q0_var[94] <= 2157.17943612574
 Q0_var[95] <= 2157.17943612574
 Q0_var[96] <= 2157.17943612574
 Q0_var[97] <= 2157.17943612574
 Q0_var[98] <= 2157.17943612574
 Q0_var[99] <= 2157.17943612574
 Q0_var[100] <= 2157.17943612574
 Q0_var[101] <= 2157.17943612574
 Q0_var[102] <= 2157.17943612574
 Q0_var[103] <= 2157.17943612574
 Q0_var[104] <= 2157.17943612574
 Q0_var[105] <= 2157.17943612574
 Q0_var[106] <= 2157.17943612574
 Q0_var[107] <= 2157.17943612574
 Q0_var[108] <= 2157.17943612574
 Q0_var[109] <= 2157.17943612574
 Q0_var[110] <= 2157.17943612574
 Q0_var[111] <= 2157.17943612574
 Q0_var[112] <= 2157.17943612574
 Q0_var[113] <= 2157.17943612574
 Q0_var[114] <= 2157.17943612574
 Q0_var[115] <= 2157.17943612574
 Q0_var[116] <= 2157.17943612574
 Q0_var[117] <= 2157.17943612574
 Q0_var[118] <= 2157.17943612574
 Q0_var[119] <= 2157.17943612574
 Q0_var[120] <= 2157.17943612574
 Q0_var[121] <= 2157.17943612574
 Q0_var[122] <= 2157.17943612574
 Q0_var[123] <= 2157.17943612574
 Q0_var[124] <= 2157.17943612574
 Q0_var[125] <= 2157.17943612574
 Q0_var[126] <= 2157.17943612574
 Q0_var[127] <= 2157.17943612574
 Q0_var[128] <= 2157.17943612574
 Q0_var[129] <= 2157.17943612574
 Q0_var[130] <= 2157.17943612574
 Q0_var[131] <= 2157.17943612574
 Q0_var[132] <= 2157.17943612574
 Q0_var[133] <= 2157.17943612574
 Q0_var[134] <= 2157.17943612574
 Q0_var[135] <= 2157.17943612574
 Q0_var[136] <= 2157.17943612574
 Q0_var[137] <= 2157.17943612574
 Q0_var[138] <= 2157.17943612574
 Q0_var[139] <= 2157.17943612574
 Q0_var[140] <= 2157.17943612574
 Q0_var[141] <= 2157.17943612574
 Q0_var[142] <= 2157.17943612574
 Q0_var[143] <= 2157.17943612574
 Q0_var[144] <= 2157.17943612574
 Q0_var[145] <= 2157.17943612574
 Q0_var[146] <= 2157.17943612574
 Q0_var[147] <= 2157.17943612574
 Q0_var[148] <= 2157.17943612574
 Q0_var[149] <= 2157.17943612574
 Q0_var[150] <= 2157.17943612574
 Q0_var[151] <= 2157.17943612574
 Q0_var[152] <= 2157.17943612574
 Q0_var[153] <= 2157.17943612574
 Q0_var[154] <= 2157.17943612574
 Q0_var[155] <= 2157.17943612574
 Q0_var[156] <= 2157.17943612574
 Q0_var[157] <= 2157.17943612574
 Q0_var[158] <= 2157.17943612574
 Q0_var[159] <= 2157.17943612574
 Q0_var[160] <= 2157.17943612574
 Q0_var[161] <= 2157.17943612574
 Q0_var[162] <= 2157.17943612574
 Q0_var[163] <= 2157.17943612574
 Q0_var[164] <= 2157.17943612574
 Q0_var[165] <= 2157.17943612574
 Q0_var[166] <= 2157.17943612574
 Q0_var[167] <= 2157.17943612574
 Q0_var[168] <= 2157.17943612574
 Q0_var[169] <= 2157.17943612574
 Q0_var[170] <= 2157.17943612574
 Q0_var[171] <= 2157.17943612574
 Q0_var[172] <= 2157.17943612574
 Q0_var[173] <= 2157.17943612574
 Q0_var[174] <= 2157.17943612574
 Q0_var[175] <= 2157.17943612574
 Q0_var[176] <= 2157.17943612574
 Q0_var[177] <= 2157.17943612574
 Q0_var[178] <= 2157.17943612574
 Q0_var[179] <= 2157.17943612574
 Q0_var[180] <= 2157.17943612574
 Q0_var[181] <= 2157.17943612574
 Q0_var[182] <= 2157.17943612574
 Q0_var[183] <= 2157.17943612574
 Q0_var[184] <= 2157.17943612574
 Q0_var[185] <= 2157.17943612574
 Q0_var[186] <= 2157.17943612574
 Q0_var[187] <= 2157.17943612574
 Q0_var[188] <= 2157.17943612574
 Q0_var[189] <= 2157.17943612574
 Q0_var[190] <= 2157.17943612574
 Q0_var[191] <= 2157.17943612574
 Q0_var[192] <= 2157.17943612574
 Q0_var[193] <= 2157.17943612574
 Q0_var[194] <= 2157.17943612574
 Q0_var[195] <= 2157.17943612574
 Q0_var[196] <= 2157.17943612574
 Q0_var[197] <= 2157.17943612574
 Q0_var[198] <= 2157.17943612574
 Q0_var[199] <= 2157.17943612574
 Q0_var[200] <= 2157.17943612574
 Q0_var[201] <= 2157.17943612574
 Q0_var[202] <= 2157.17943612574
 Q0_var[203] <= 2157.17943612574
 Q0_var[204] <= 2157.17943612574
 Q0_var[205] <= 2157.17943612574
 Q0_var[206] <= 2157.17943612574
 Q0_var[207] <= 2157.17943612574
 Q0_var[208] <= 2157.17943612574
 Q0_var[209] <= 2157.17943612574
 Q0_var[210] <= 2157.17943612574
 Q0_var[211] <= 2157.17943612574
 Q0_var[212] <= 2157.17943612574
 Q0_var[213] <= 2157.17943612574
 Q0_var[214] <= 2157.17943612574
 Q0_var[215] <= 2157.17943612574
 Q0_var[216] <= 2157.17943612574
 Q0_var[217] <= 2157.17943612574
 Q0_var[218] <= 2157.17943612574
 Q0_var[219] <= 2157.17943612574
 Q0_var[220] <= 2157.17943612574
 Q0_var[221] <= 2157.17943612574
 Q0_var[222] <= 2157.17943612574
 Q0_var[223] <= 2157.17943612574
 Q0_var[224] <= 2157.17943612574
 Q0_var[225] <= 2157.17943612574
 Q0_var[226] <= 2157.17943612574
 Q0_var[227] <= 2157.17943612574
 Q0_var[228] <= 2157.17943612574
 Q0_var[229] <= 2157.17943612574
 Q0_var[230] <= 2157.17943612574
 Q0_var[231] <= 2157.17943612574
 Q0_var[232] <= 2157.17943612574
 Q0_var[233] <= 2157.17943612574
 Q0_var[234] <= 2157.17943612574
 Q0_var[235] <= 2157.17943612574
 Q0_var[236] <= 2157.17943612574
 Q0_var[237] <= 2157.17943612574
 Q0_var[238] <= 2157.17943612574
 Q0_var[239] <= 2157.17943612574
 Q0_var[240] <= 2157.17943612574
 Q0_var[241] <= 2157.17943612574
 Q0_var[242] <= 2157.17943612574
 Q0_var[243] <= 2157.17943612574
 Q0_var[244] <= 2157.17943612574
 Q0_var[245] <= 2157.17943612574
 Q0_var[246] <= 2157.17943612574
 Q0_var[247] <= 2157.17943612574
 Q0_var[248] <= 2157.17943612574
 Q0_var[249] <= 2157.17943612574
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
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 311.0251636130733 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 1562.819790626047 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 105.9702103844509 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 324.5138204310281 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 534.4887839207412 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 974.5514912467027 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 459.6870818321154 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 1886.664143052687 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 274.6221011453696 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 825.6712004476807 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 622.1284308325011 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 2092.033355181434 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 417.1972146883232 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 1471.450587105507 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 57.70010585339908 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 379.7910369710239 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 304.7706936787965 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 619.0432990857677 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 655.0289795422884 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 2150.174818704398 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 276.3227474035778 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 2075.190483876526 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 119.5458267768758 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 928.2942621173393 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 175.550333593017 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 1120.281080114851 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 421.8738478845578 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 1319.880339667429 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 290.0192850593794 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 308.4142764821366 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 383.6992953883416 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 1158.411768917133 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 354.6680610536623 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 1164.255632456589 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 110.9118922780958 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 862.3794294522688 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 392.2528102253823 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 1666.377769428032 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 571.6462479792019 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 1729.71322047711 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 203.8340100004 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 712.982219600498 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 278.7784712176002 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 1624.237131575594 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 476.0153771204417 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 1767.007096298152 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 672.6244800796422 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 2348.230435223977 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 518.4956700972725 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 1959.719550616583 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 473.3862630798172 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 2131.774750330777 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 401.99349531143 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 1485.560539390864 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 96.42274920631569 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 939.7428894940106 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 707.4595453920876 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 2407.558964246491 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 82.32632633032061 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 462.9703976355289 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 487.8714369750719 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 1068.460843962106 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 214.6246109264699 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 582.6870558914214 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 541.7519399273892 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 2291.582259291944 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 366.5223049134499 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 2488.059168933316 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 375.6855613206947 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 2049.738536111399 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 185.5582342339428 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 1017.892244713687 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 130.7781133225293 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 856.1232990882058 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 134.2424139460701 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 920.3682126844783 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 405.8293392306349 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 819.673390786543 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 299.5267134285999 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 1542.059211743204 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 103.363624857212 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 735.0206445377808 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 440.4359615854938 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 806.7483530363293 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 478.1457423538777 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 1963.242512356521 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 390.4136837566275 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 1870.743024037578 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 138.5934337217769 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 532.6260551433593 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 401.9114084672532 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 1033.406295765255 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 451.0708956474153 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 1824.456664841893 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 701.632369255298 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 1919.353389766453 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 375.004082192915 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 1399.854312327068 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 676.143265321571 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 2042.483713405973 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
 GC300: Fraction[50] = LOGISTIC ( f_var[50] )
 min_sales_constr_50: Sold_0[50] = MIN ( Q0_var[50] , 130.7830838166814 )
 max_lost_constr_50: Lost_0[50] = MAX ( Lost_0_expr_50 , 0 )
 max_Q1_constr_50: Q1_var[50] = MAX ( Q_hat_adjusted_50 , 0 )
 min_sales2_constr_50: Sold_1[50] = MIN ( total_stock_second_period_50
   , 836.2830875038119 )
 max_lost2_constr_50: Lost_1[50] = MAX ( Lost_1_expr_50 , 0 )
 GC306: Fraction[51] = LOGISTIC ( f_var[51] )
 min_sales_constr_51: Sold_0[51] = MIN ( Q0_var[51] , 104.2038153379804 )
 max_lost_constr_51: Lost_0[51] = MAX ( Lost_0_expr_51 , 0 )
 max_Q1_constr_51: Q1_var[51] = MAX ( Q_hat_adjusted_51 , 0 )
 min_sales2_constr_51: Sold_1[51] = MIN ( total_stock_second_period_51
   , 597.627207980269 )
 max_lost2_constr_51: Lost_1[51] = MAX ( Lost_1_expr_51 , 0 )
 GC312: Fraction[52] = LOGISTIC ( f_var[52] )
 min_sales_constr_52: Sold_0[52] = MIN ( Q0_var[52] , 415.4612156881613 )
 max_lost_constr_52: Lost_0[52] = MAX ( Lost_0_expr_52 , 0 )
 max_Q1_constr_52: Q1_var[52] = MAX ( Q_hat_adjusted_52 , 0 )
 min_sales2_constr_52: Sold_1[52] = MIN ( total_stock_second_period_52
   , 1610.044917623197 )
 max_lost2_constr_52: Lost_1[52] = MAX ( Lost_1_expr_52 , 0 )
 GC318: Fraction[53] = LOGISTIC ( f_var[53] )
 min_sales_constr_53: Sold_0[53] = MIN ( Q0_var[53] , 303.3280717677706 )
 max_lost_constr_53: Lost_0[53] = MAX ( Lost_0_expr_53 , 0 )
 max_Q1_constr_53: Q1_var[53] = MAX ( Q_hat_adjusted_53 , 0 )
 min_sales2_constr_53: Sold_1[53] = MIN ( total_stock_second_period_53
   , 761.4476958833377 )
 max_lost2_constr_53: Lost_1[53] = MAX ( Lost_1_expr_53 , 0 )
 GC324: Fraction[54] = LOGISTIC ( f_var[54] )
 min_sales_constr_54: Sold_0[54] = MIN ( Q0_var[54] , 493.1986700210101 )
 max_lost_constr_54: Lost_0[54] = MAX ( Lost_0_expr_54 , 0 )
 max_Q1_constr_54: Q1_var[54] = MAX ( Q_hat_adjusted_54 , 0 )
 min_sales2_constr_54: Sold_1[54] = MIN ( total_stock_second_period_54
   , 2247.345197135545 )
 max_lost2_constr_54: Lost_1[54] = MAX ( Lost_1_expr_54 , 0 )
 GC330: Fraction[55] = LOGISTIC ( f_var[55] )
 min_sales_constr_55: Sold_0[55] = MIN ( Q0_var[55] , 133.8092946906124 )
 max_lost_constr_55: Lost_0[55] = MAX ( Lost_0_expr_55 , 0 )
 max_Q1_constr_55: Q1_var[55] = MAX ( Q_hat_adjusted_55 , 0 )
 min_sales2_constr_55: Sold_1[55] = MIN ( total_stock_second_period_55
   , 775.3455395011633 )
 max_lost2_constr_55: Lost_1[55] = MAX ( Lost_1_expr_55 , 0 )
 GC336: Fraction[56] = LOGISTIC ( f_var[56] )
 min_sales_constr_56: Sold_0[56] = MIN ( Q0_var[56] , 127.655010513551 )
 max_lost_constr_56: Lost_0[56] = MAX ( Lost_0_expr_56 , 0 )
 max_Q1_constr_56: Q1_var[56] = MAX ( Q_hat_adjusted_56 , 0 )
 min_sales2_constr_56: Sold_1[56] = MIN ( total_stock_second_period_56
   , 1204.369459325605 )
 max_lost2_constr_56: Lost_1[56] = MAX ( Lost_1_expr_56 , 0 )
 GC342: Fraction[57] = LOGISTIC ( f_var[57] )
 min_sales_constr_57: Sold_0[57] = MIN ( Q0_var[57] , 436.73253927975 )
 max_lost_constr_57: Lost_0[57] = MAX ( Lost_0_expr_57 , 0 )
 max_Q1_constr_57: Q1_var[57] = MAX ( Q_hat_adjusted_57 , 0 )
 min_sales2_constr_57: Sold_1[57] = MIN ( total_stock_second_period_57
   , 2157.210000812454 )
 max_lost2_constr_57: Lost_1[57] = MAX ( Lost_1_expr_57 , 0 )
 GC348: Fraction[58] = LOGISTIC ( f_var[58] )
 min_sales_constr_58: Sold_0[58] = MIN ( Q0_var[58] , 643.537895569871 )
 max_lost_constr_58: Lost_0[58] = MAX ( Lost_0_expr_58 , 0 )
 max_Q1_constr_58: Q1_var[58] = MAX ( Q_hat_adjusted_58 , 0 )
 min_sales2_constr_58: Sold_1[58] = MIN ( total_stock_second_period_58
   , 1907.556576283753 )
 max_lost2_constr_58: Lost_1[58] = MAX ( Lost_1_expr_58 , 0 )
 GC354: Fraction[59] = LOGISTIC ( f_var[59] )
 min_sales_constr_59: Sold_0[59] = MIN ( Q0_var[59] , 353.7419563365844 )
 max_lost_constr_59: Lost_0[59] = MAX ( Lost_0_expr_59 , 0 )
 max_Q1_constr_59: Q1_var[59] = MAX ( Q_hat_adjusted_59 , 0 )
 min_sales2_constr_59: Sold_1[59] = MIN ( total_stock_second_period_59
   , 1514.763625437006 )
 max_lost2_constr_59: Lost_1[59] = MAX ( Lost_1_expr_59 , 0 )
 GC360: Fraction[60] = LOGISTIC ( f_var[60] )
 min_sales_constr_60: Sold_0[60] = MIN ( Q0_var[60] , 598.9061411100495 )
 max_lost_constr_60: Lost_0[60] = MAX ( Lost_0_expr_60 , 0 )
 max_Q1_constr_60: Q1_var[60] = MAX ( Q_hat_adjusted_60 , 0 )
 min_sales2_constr_60: Sold_1[60] = MIN ( total_stock_second_period_60
   , 1834.376561371839 )
 max_lost2_constr_60: Lost_1[60] = MAX ( Lost_1_expr_60 , 0 )
 GC366: Fraction[61] = LOGISTIC ( f_var[61] )
 min_sales_constr_61: Sold_0[61] = MIN ( Q0_var[61] , 377.1809117046377 )
 max_lost_constr_61: Lost_0[61] = MAX ( Lost_0_expr_61 , 0 )
 max_Q1_constr_61: Q1_var[61] = MAX ( Q_hat_adjusted_61 , 0 )
 min_sales2_constr_61: Sold_1[61] = MIN ( total_stock_second_period_61
   , 1212.213327660204 )
 max_lost2_constr_61: Lost_1[61] = MAX ( Lost_1_expr_61 , 0 )
 GC372: Fraction[62] = LOGISTIC ( f_var[62] )
 min_sales_constr_62: Sold_0[62] = MIN ( Q0_var[62] , 56.09568451910415 )
 max_lost_constr_62: Lost_0[62] = MAX ( Lost_0_expr_62 , 0 )
 max_Q1_constr_62: Q1_var[62] = MAX ( Q_hat_adjusted_62 , 0 )
 min_sales2_constr_62: Sold_1[62] = MIN ( total_stock_second_period_62
   , 539.200158408942 )
 max_lost2_constr_62: Lost_1[62] = MAX ( Lost_1_expr_62 , 0 )
 GC378: Fraction[63] = LOGISTIC ( f_var[63] )
 min_sales_constr_63: Sold_0[63] = MIN ( Q0_var[63] , 109.074867204533 )
 max_lost_constr_63: Lost_0[63] = MAX ( Lost_0_expr_63 , 0 )
 max_Q1_constr_63: Q1_var[63] = MAX ( Q_hat_adjusted_63 , 0 )
 min_sales2_constr_63: Sold_1[63] = MIN ( total_stock_second_period_63
   , 353.153108878996 )
 max_lost2_constr_63: Lost_1[63] = MAX ( Lost_1_expr_63 , 0 )
 GC384: Fraction[64] = LOGISTIC ( f_var[64] )
 min_sales_constr_64: Sold_0[64] = MIN ( Q0_var[64] , 296.4050669163157 )
 max_lost_constr_64: Lost_0[64] = MAX ( Lost_0_expr_64 , 0 )
 max_Q1_constr_64: Q1_var[64] = MAX ( Q_hat_adjusted_64 , 0 )
 min_sales2_constr_64: Sold_1[64] = MIN ( total_stock_second_period_64
   , 1202.570418835342 )
 max_lost2_constr_64: Lost_1[64] = MAX ( Lost_1_expr_64 , 0 )
 GC390: Fraction[65] = LOGISTIC ( f_var[65] )
 min_sales_constr_65: Sold_0[65] = MIN ( Q0_var[65] , 70.72097073381131 )
 max_lost_constr_65: Lost_0[65] = MAX ( Lost_0_expr_65 , 0 )
 max_Q1_constr_65: Q1_var[65] = MAX ( Q_hat_adjusted_65 , 0 )
 min_sales2_constr_65: Sold_1[65] = MIN ( total_stock_second_period_65
   , 808.7000449059769 )
 max_lost2_constr_65: Lost_1[65] = MAX ( Lost_1_expr_65 , 0 )
 GC396: Fraction[66] = LOGISTIC ( f_var[66] )
 min_sales_constr_66: Sold_0[66] = MIN ( Q0_var[66] , 124.2284859013619 )
 max_lost_constr_66: Lost_0[66] = MAX ( Lost_0_expr_66 , 0 )
 max_Q1_constr_66: Q1_var[66] = MAX ( Q_hat_adjusted_66 , 0 )
 min_sales2_constr_66: Sold_1[66] = MIN ( total_stock_second_period_66
   , 567.8551966448016 )
 max_lost2_constr_66: Lost_1[66] = MAX ( Lost_1_expr_66 , 0 )
 GC402: Fraction[67] = LOGISTIC ( f_var[67] )
 min_sales_constr_67: Sold_0[67] = MIN ( Q0_var[67] , 665.7984071163157 )
 max_lost_constr_67: Lost_0[67] = MAX ( Lost_0_expr_67 , 0 )
 max_Q1_constr_67: Q1_var[67] = MAX ( Q_hat_adjusted_67 , 0 )
 min_sales2_constr_67: Sold_1[67] = MIN ( total_stock_second_period_67
   , 2660.793811367604 )
 max_lost2_constr_67: Lost_1[67] = MAX ( Lost_1_expr_67 , 0 )
 GC408: Fraction[68] = LOGISTIC ( f_var[68] )
 min_sales_constr_68: Sold_0[68] = MIN ( Q0_var[68] , 561.684926219236 )
 max_lost_constr_68: Lost_0[68] = MAX ( Lost_0_expr_68 , 0 )
 max_Q1_constr_68: Q1_var[68] = MAX ( Q_hat_adjusted_68 , 0 )
 min_sales2_constr_68: Sold_1[68] = MIN ( total_stock_second_period_68
   , 1926.346080803945 )
 max_lost2_constr_68: Lost_1[68] = MAX ( Lost_1_expr_68 , 0 )
 GC414: Fraction[69] = LOGISTIC ( f_var[69] )
 min_sales_constr_69: Sold_0[69] = MIN ( Q0_var[69] , 218.0246496293819 )
 max_lost_constr_69: Lost_0[69] = MAX ( Lost_0_expr_69 , 0 )
 max_Q1_constr_69: Q1_var[69] = MAX ( Q_hat_adjusted_69 , 0 )
 min_sales2_constr_69: Sold_1[69] = MIN ( total_stock_second_period_69
   , 833.1122346048919 )
 max_lost2_constr_69: Lost_1[69] = MAX ( Lost_1_expr_69 , 0 )
 GC420: Fraction[70] = LOGISTIC ( f_var[70] )
 min_sales_constr_70: Sold_0[70] = MIN ( Q0_var[70] , 236.213596691663 )
 max_lost_constr_70: Lost_0[70] = MAX ( Lost_0_expr_70 , 0 )
 max_Q1_constr_70: Q1_var[70] = MAX ( Q_hat_adjusted_70 , 0 )
 min_sales2_constr_70: Sold_1[70] = MIN ( total_stock_second_period_70
   , 583.9643147548329 )
 max_lost2_constr_70: Lost_1[70] = MAX ( Lost_1_expr_70 , 0 )
 GC426: Fraction[71] = LOGISTIC ( f_var[71] )
 min_sales_constr_71: Sold_0[71] = MIN ( Q0_var[71] , 706.6521279301862 )
 max_lost_constr_71: Lost_0[71] = MAX ( Lost_0_expr_71 , 0 )
 max_Q1_constr_71: Q1_var[71] = MAX ( Q_hat_adjusted_71 , 0 )
 min_sales2_constr_71: Sold_1[71] = MIN ( total_stock_second_period_71
   , 2189.636400534236 )
 max_lost2_constr_71: Lost_1[71] = MAX ( Lost_1_expr_71 , 0 )
 GC432: Fraction[72] = LOGISTIC ( f_var[72] )
 min_sales_constr_72: Sold_0[72] = MIN ( Q0_var[72] , 196.8179581575923 )
 max_lost_constr_72: Lost_0[72] = MAX ( Lost_0_expr_72 , 0 )
 max_Q1_constr_72: Q1_var[72] = MAX ( Q_hat_adjusted_72 , 0 )
 min_sales2_constr_72: Sold_1[72] = MIN ( total_stock_second_period_72
   , 1006.990015484635 )
 max_lost2_constr_72: Lost_1[72] = MAX ( Lost_1_expr_72 , 0 )
 GC438: Fraction[73] = LOGISTIC ( f_var[73] )
 min_sales_constr_73: Sold_0[73] = MIN ( Q0_var[73] , 276.246735988532 )
 max_lost_constr_73: Lost_0[73] = MAX ( Lost_0_expr_73 , 0 )
 max_Q1_constr_73: Q1_var[73] = MAX ( Q_hat_adjusted_73 , 0 )
 min_sales2_constr_73: Sold_1[73] = MIN ( total_stock_second_period_73
   , 724.1258080325795 )
 max_lost2_constr_73: Lost_1[73] = MAX ( Lost_1_expr_73 , 0 )
 GC444: Fraction[74] = LOGISTIC ( f_var[74] )
 min_sales_constr_74: Sold_0[74] = MIN ( Q0_var[74] , 380.0599411317962 )
 max_lost_constr_74: Lost_0[74] = MAX ( Lost_0_expr_74 , 0 )
 max_Q1_constr_74: Q1_var[74] = MAX ( Q_hat_adjusted_74 , 0 )
 min_sales2_constr_74: Sold_1[74] = MIN ( total_stock_second_period_74
   , 1582.648076436338 )
 max_lost2_constr_74: Lost_1[74] = MAX ( Lost_1_expr_74 , 0 )
 GC450: Fraction[75] = LOGISTIC ( f_var[75] )
 min_sales_constr_75: Sold_0[75] = MIN ( Q0_var[75] , 77.12931336098755 )
 max_lost_constr_75: Lost_0[75] = MAX ( Lost_0_expr_75 , 0 )
 max_Q1_constr_75: Q1_var[75] = MAX ( Q_hat_adjusted_75 , 0 )
 min_sales2_constr_75: Sold_1[75] = MIN ( total_stock_second_period_75
   , 817.9594699824568 )
 max_lost2_constr_75: Lost_1[75] = MAX ( Lost_1_expr_75 , 0 )
 GC456: Fraction[76] = LOGISTIC ( f_var[76] )
 min_sales_constr_76: Sold_0[76] = MIN ( Q0_var[76] , 250.1657757249098 )
 max_lost_constr_76: Lost_0[76] = MAX ( Lost_0_expr_76 , 0 )
 max_Q1_constr_76: Q1_var[76] = MAX ( Q_hat_adjusted_76 , 0 )
 min_sales2_constr_76: Sold_1[76] = MIN ( total_stock_second_period_76
   , 743.2452575472593 )
 max_lost2_constr_76: Lost_1[76] = MAX ( Lost_1_expr_76 , 0 )
 GC462: Fraction[77] = LOGISTIC ( f_var[77] )
 min_sales_constr_77: Sold_0[77] = MIN ( Q0_var[77] , 669.803650277536 )
 max_lost_constr_77: Lost_0[77] = MAX ( Lost_0_expr_77 , 0 )
 max_Q1_constr_77: Q1_var[77] = MAX ( Q_hat_adjusted_77 , 0 )
 min_sales2_constr_77: Sold_1[77] = MIN ( total_stock_second_period_77
   , 2445.191357157138 )
 max_lost2_constr_77: Lost_1[77] = MAX ( Lost_1_expr_77 , 0 )
 GC468: Fraction[78] = LOGISTIC ( f_var[78] )
 min_sales_constr_78: Sold_0[78] = MIN ( Q0_var[78] , 280.7808026233366 )
 max_lost_constr_78: Lost_0[78] = MAX ( Lost_0_expr_78 , 0 )
 max_Q1_constr_78: Q1_var[78] = MAX ( Q_hat_adjusted_78 , 0 )
 min_sales2_constr_78: Sold_1[78] = MIN ( total_stock_second_period_78
   , 1633.052065568451 )
 max_lost2_constr_78: Lost_1[78] = MAX ( Lost_1_expr_78 , 0 )
 GC474: Fraction[79] = LOGISTIC ( f_var[79] )
 min_sales_constr_79: Sold_0[79] = MIN ( Q0_var[79] , 465.9637668013111 )
 max_lost_constr_79: Lost_0[79] = MAX ( Lost_0_expr_79 , 0 )
 max_Q1_constr_79: Q1_var[79] = MAX ( Q_hat_adjusted_79 , 0 )
 min_sales2_constr_79: Sold_1[79] = MIN ( total_stock_second_period_79
   , 1564.405145323783 )
 max_lost2_constr_79: Lost_1[79] = MAX ( Lost_1_expr_79 , 0 )
 GC480: Fraction[80] = LOGISTIC ( f_var[80] )
 min_sales_constr_80: Sold_0[80] = MIN ( Q0_var[80] , 238.4104139152425 )
 max_lost_constr_80: Lost_0[80] = MAX ( Lost_0_expr_80 , 0 )
 max_Q1_constr_80: Q1_var[80] = MAX ( Q_hat_adjusted_80 , 0 )
 min_sales2_constr_80: Sold_1[80] = MIN ( total_stock_second_period_80
   , 1239.036119925152 )
 max_lost2_constr_80: Lost_1[80] = MAX ( Lost_1_expr_80 , 0 )
 GC486: Fraction[81] = LOGISTIC ( f_var[81] )
 min_sales_constr_81: Sold_0[81] = MIN ( Q0_var[81] , 487.4757754160669 )
 max_lost_constr_81: Lost_0[81] = MAX ( Lost_0_expr_81 , 0 )
 max_Q1_constr_81: Q1_var[81] = MAX ( Q_hat_adjusted_81 , 0 )
 min_sales2_constr_81: Sold_1[81] = MIN ( total_stock_second_period_81
   , 1684.120878757362 )
 max_lost2_constr_81: Lost_1[81] = MAX ( Lost_1_expr_81 , 0 )
 GC492: Fraction[82] = LOGISTIC ( f_var[82] )
 min_sales_constr_82: Sold_0[82] = MIN ( Q0_var[82] , 191.4951348800481 )
 max_lost_constr_82: Lost_0[82] = MAX ( Lost_0_expr_82 , 0 )
 max_Q1_constr_82: Q1_var[82] = MAX ( Q_hat_adjusted_82 , 0 )
 min_sales2_constr_82: Sold_1[82] = MIN ( total_stock_second_period_82
   , 805.6324794119407 )
 max_lost2_constr_82: Lost_1[82] = MAX ( Lost_1_expr_82 , 0 )
 GC498: Fraction[83] = LOGISTIC ( f_var[83] )
 min_sales_constr_83: Sold_0[83] = MIN ( Q0_var[83] , 586.8433741013578 )
 max_lost_constr_83: Lost_0[83] = MAX ( Lost_0_expr_83 , 0 )
 max_Q1_constr_83: Q1_var[83] = MAX ( Q_hat_adjusted_83 , 0 )
 min_sales2_constr_83: Sold_1[83] = MIN ( total_stock_second_period_83
   , 2272.263269995019 )
 max_lost2_constr_83: Lost_1[83] = MAX ( Lost_1_expr_83 , 0 )
 GC504: Fraction[84] = LOGISTIC ( f_var[84] )
 min_sales_constr_84: Sold_0[84] = MIN ( Q0_var[84] , 496.4411566341943 )
 max_lost_constr_84: Lost_0[84] = MAX ( Lost_0_expr_84 , 0 )
 max_Q1_constr_84: Q1_var[84] = MAX ( Q_hat_adjusted_84 , 0 )
 min_sales2_constr_84: Sold_1[84] = MIN ( total_stock_second_period_84
   , 2075.844551681394 )
 max_lost2_constr_84: Lost_1[84] = MAX ( Lost_1_expr_84 , 0 )
 GC510: Fraction[85] = LOGISTIC ( f_var[85] )
 min_sales_constr_85: Sold_0[85] = MIN ( Q0_var[85] , 532.8488618728193 )
 max_lost_constr_85: Lost_0[85] = MAX ( Lost_0_expr_85 , 0 )
 max_Q1_constr_85: Q1_var[85] = MAX ( Q_hat_adjusted_85 , 0 )
 min_sales2_constr_85: Sold_1[85] = MIN ( total_stock_second_period_85
   , 1610.547469894234 )
 max_lost2_constr_85: Lost_1[85] = MAX ( Lost_1_expr_85 , 0 )
 GC516: Fraction[86] = LOGISTIC ( f_var[86] )
 min_sales_constr_86: Sold_0[86] = MIN ( Q0_var[86] , 60.70907708378834 )
 max_lost_constr_86: Lost_0[86] = MAX ( Lost_0_expr_86 , 0 )
 max_Q1_constr_86: Q1_var[86] = MAX ( Q_hat_adjusted_86 , 0 )
 min_sales2_constr_86: Sold_1[86] = MIN ( total_stock_second_period_86
   , 608.1999388661843 )
 max_lost2_constr_86: Lost_1[86] = MAX ( Lost_1_expr_86 , 0 )
 GC522: Fraction[87] = LOGISTIC ( f_var[87] )
 min_sales_constr_87: Sold_0[87] = MIN ( Q0_var[87] , 369.1660006900256 )
 max_lost_constr_87: Lost_0[87] = MAX ( Lost_0_expr_87 , 0 )
 max_Q1_constr_87: Q1_var[87] = MAX ( Q_hat_adjusted_87 , 0 )
 min_sales2_constr_87: Sold_1[87] = MIN ( total_stock_second_period_87
   , 1926.039820860612 )
 max_lost2_constr_87: Lost_1[87] = MAX ( Lost_1_expr_87 , 0 )
 GC528: Fraction[88] = LOGISTIC ( f_var[88] )
 min_sales_constr_88: Sold_0[88] = MIN ( Q0_var[88] , 407.9740012446229 )
 max_lost_constr_88: Lost_0[88] = MAX ( Lost_0_expr_88 , 0 )
 max_Q1_constr_88: Q1_var[88] = MAX ( Q_hat_adjusted_88 , 0 )
 min_sales2_constr_88: Sold_1[88] = MIN ( total_stock_second_period_88
   , 2441.950956662425 )
 max_lost2_constr_88: Lost_1[88] = MAX ( Lost_1_expr_88 , 0 )
 GC534: Fraction[89] = LOGISTIC ( f_var[89] )
 min_sales_constr_89: Sold_0[89] = MIN ( Q0_var[89] , 333.2553660523799 )
 max_lost_constr_89: Lost_0[89] = MAX ( Lost_0_expr_89 , 0 )
 max_Q1_constr_89: Q1_var[89] = MAX ( Q_hat_adjusted_89 , 0 )
 min_sales2_constr_89: Sold_1[89] = MIN ( total_stock_second_period_89
   , 1296.321948003394 )
 max_lost2_constr_89: Lost_1[89] = MAX ( Lost_1_expr_89 , 0 )
 GC540: Fraction[90] = LOGISTIC ( f_var[90] )
 min_sales_constr_90: Sold_0[90] = MIN ( Q0_var[90] , 325.5925592186188 )
 max_lost_constr_90: Lost_0[90] = MAX ( Lost_0_expr_90 , 0 )
 max_Q1_constr_90: Q1_var[90] = MAX ( Q_hat_adjusted_90 , 0 )
 min_sales2_constr_90: Sold_1[90] = MIN ( total_stock_second_period_90
   , 1035.901823795766 )
 max_lost2_constr_90: Lost_1[90] = MAX ( Lost_1_expr_90 , 0 )
 GC546: Fraction[91] = LOGISTIC ( f_var[91] )
 min_sales_constr_91: Sold_0[91] = MIN ( Q0_var[91] , 80.3238758800838 )
 max_lost_constr_91: Lost_0[91] = MAX ( Lost_0_expr_91 , 0 )
 max_Q1_constr_91: Q1_var[91] = MAX ( Q_hat_adjusted_91 , 0 )
 min_sales2_constr_91: Sold_1[91] = MIN ( total_stock_second_period_91
   , 684.3430090950701 )
 max_lost2_constr_91: Lost_1[91] = MAX ( Lost_1_expr_91 , 0 )
 GC552: Fraction[92] = LOGISTIC ( f_var[92] )
 min_sales_constr_92: Sold_0[92] = MIN ( Q0_var[92] , 538.589861310528 )
 max_lost_constr_92: Lost_0[92] = MAX ( Lost_0_expr_92 , 0 )
 max_Q1_constr_92: Q1_var[92] = MAX ( Q_hat_adjusted_92 , 0 )
 min_sales2_constr_92: Sold_1[92] = MIN ( total_stock_second_period_92
   , 1525.291727435214 )
 max_lost2_constr_92: Lost_1[92] = MAX ( Lost_1_expr_92 , 0 )
 GC558: Fraction[93] = LOGISTIC ( f_var[93] )
 min_sales_constr_93: Sold_0[93] = MIN ( Q0_var[93] , 371.0826538671789 )
 max_lost_constr_93: Lost_0[93] = MAX ( Lost_0_expr_93 , 0 )
 max_Q1_constr_93: Q1_var[93] = MAX ( Q_hat_adjusted_93 , 0 )
 min_sales2_constr_93: Sold_1[93] = MIN ( total_stock_second_period_93
   , 2271.819961610808 )
 max_lost2_constr_93: Lost_1[93] = MAX ( Lost_1_expr_93 , 0 )
 GC564: Fraction[94] = LOGISTIC ( f_var[94] )
 min_sales_constr_94: Sold_0[94] = MIN ( Q0_var[94] , 181.0184262136766 )
 max_lost_constr_94: Lost_0[94] = MAX ( Lost_0_expr_94 , 0 )
 max_Q1_constr_94: Q1_var[94] = MAX ( Q_hat_adjusted_94 , 0 )
 min_sales2_constr_94: Sold_1[94] = MIN ( total_stock_second_period_94
   , 641.591686240127 )
 max_lost2_constr_94: Lost_1[94] = MAX ( Lost_1_expr_94 , 0 )
 GC570: Fraction[95] = LOGISTIC ( f_var[95] )
 min_sales_constr_95: Sold_0[95] = MIN ( Q0_var[95] , 482.5269160686618 )
 max_lost_constr_95: Lost_0[95] = MAX ( Lost_0_expr_95 , 0 )
 max_Q1_constr_95: Q1_var[95] = MAX ( Q_hat_adjusted_95 , 0 )
 min_sales2_constr_95: Sold_1[95] = MIN ( total_stock_second_period_95
   , 1603.177986769715 )
 max_lost2_constr_95: Lost_1[95] = MAX ( Lost_1_expr_95 , 0 )
 GC576: Fraction[96] = LOGISTIC ( f_var[96] )
 min_sales_constr_96: Sold_0[96] = MIN ( Q0_var[96] , 147.7291206207728 )
 max_lost_constr_96: Lost_0[96] = MAX ( Lost_0_expr_96 , 0 )
 max_Q1_constr_96: Q1_var[96] = MAX ( Q_hat_adjusted_96 , 0 )
 min_sales2_constr_96: Sold_1[96] = MIN ( total_stock_second_period_96
   , 402.2032392178372 )
 max_lost2_constr_96: Lost_1[96] = MAX ( Lost_1_expr_96 , 0 )
 GC582: Fraction[97] = LOGISTIC ( f_var[97] )
 min_sales_constr_97: Sold_0[97] = MIN ( Q0_var[97] , 719.3519782140543 )
 max_lost_constr_97: Lost_0[97] = MAX ( Lost_0_expr_97 , 0 )
 max_Q1_constr_97: Q1_var[97] = MAX ( Q_hat_adjusted_97 , 0 )
 min_sales2_constr_97: Sold_1[97] = MIN ( total_stock_second_period_97
   , 2106.99266033027 )
 max_lost2_constr_97: Lost_1[97] = MAX ( Lost_1_expr_97 , 0 )
 GC588: Fraction[98] = LOGISTIC ( f_var[98] )
 min_sales_constr_98: Sold_0[98] = MIN ( Q0_var[98] , 393.7542105669801 )
 max_lost_constr_98: Lost_0[98] = MAX ( Lost_0_expr_98 , 0 )
 max_Q1_constr_98: Q1_var[98] = MAX ( Q_hat_adjusted_98 , 0 )
 min_sales2_constr_98: Sold_1[98] = MIN ( total_stock_second_period_98
   , 1785.49396348099 )
 max_lost2_constr_98: Lost_1[98] = MAX ( Lost_1_expr_98 , 0 )
 GC594: Fraction[99] = LOGISTIC ( f_var[99] )
 min_sales_constr_99: Sold_0[99] = MIN ( Q0_var[99] , 405.1130514348309 )
 max_lost_constr_99: Lost_0[99] = MAX ( Lost_0_expr_99 , 0 )
 max_Q1_constr_99: Q1_var[99] = MAX ( Q_hat_adjusted_99 , 0 )
 min_sales2_constr_99: Sold_1[99] = MIN ( total_stock_second_period_99
   , 1622.786544699135 )
 max_lost2_constr_99: Lost_1[99] = MAX ( Lost_1_expr_99 , 0 )
 GC600: Fraction[100] = LOGISTIC ( f_var[100] )
 min_sales_constr_100: Sold_0[100] = MIN ( Q0_var[100] , 264.0053245797885 )
 max_lost_constr_100: Lost_0[100] = MAX ( Lost_0_expr_100 , 0 )
 max_Q1_constr_100: Q1_var[100] = MAX ( Q_hat_adjusted_100 , 0 )
 min_sales2_constr_100: Sold_1[100] = MIN ( total_stock_second_period_100
   , 1011.934648091743 )
 max_lost2_constr_100: Lost_1[100] = MAX ( Lost_1_expr_100 , 0 )
 GC606: Fraction[101] = LOGISTIC ( f_var[101] )
 min_sales_constr_101: Sold_0[101] = MIN ( Q0_var[101] , 684.0315242476323 )
 max_lost_constr_101: Lost_0[101] = MAX ( Lost_0_expr_101 , 0 )
 max_Q1_constr_101: Q1_var[101] = MAX ( Q_hat_adjusted_101 , 0 )
 min_sales2_constr_101: Sold_1[101] = MIN ( total_stock_second_period_101
   , 2084.496258901394 )
 max_lost2_constr_101: Lost_1[101] = MAX ( Lost_1_expr_101 , 0 )
 GC612: Fraction[102] = LOGISTIC ( f_var[102] )
 min_sales_constr_102: Sold_0[102] = MIN ( Q0_var[102] , 224.6611250678646 )
 max_lost_constr_102: Lost_0[102] = MAX ( Lost_0_expr_102 , 0 )
 max_Q1_constr_102: Q1_var[102] = MAX ( Q_hat_adjusted_102 , 0 )
 min_sales2_constr_102: Sold_1[102] = MIN ( total_stock_second_period_102
   , 967.1215075403429 )
 max_lost2_constr_102: Lost_1[102] = MAX ( Lost_1_expr_102 , 0 )
 GC618: Fraction[103] = LOGISTIC ( f_var[103] )
 min_sales_constr_103: Sold_0[103] = MIN ( Q0_var[103] , 590.0875281880088 )
 max_lost_constr_103: Lost_0[103] = MAX ( Lost_0_expr_103 , 0 )
 max_Q1_constr_103: Q1_var[103] = MAX ( Q_hat_adjusted_103 , 0 )
 min_sales2_constr_103: Sold_1[103] = MIN ( total_stock_second_period_103
   , 1957.951148038918 )
 max_lost2_constr_103: Lost_1[103] = MAX ( Lost_1_expr_103 , 0 )
 GC624: Fraction[104] = LOGISTIC ( f_var[104] )
 min_sales_constr_104: Sold_0[104] = MIN ( Q0_var[104] , 501.3013921662313 )
 max_lost_constr_104: Lost_0[104] = MAX ( Lost_0_expr_104 , 0 )
 max_Q1_constr_104: Q1_var[104] = MAX ( Q_hat_adjusted_104 , 0 )
 min_sales2_constr_104: Sold_1[104] = MIN ( total_stock_second_period_104
   , 2368.083219029008 )
 max_lost2_constr_104: Lost_1[104] = MAX ( Lost_1_expr_104 , 0 )
 GC630: Fraction[105] = LOGISTIC ( f_var[105] )
 min_sales_constr_105: Sold_0[105] = MIN ( Q0_var[105] , 622.251132397608 )
 max_lost_constr_105: Lost_0[105] = MAX ( Lost_0_expr_105 , 0 )
 max_Q1_constr_105: Q1_var[105] = MAX ( Q_hat_adjusted_105 , 0 )
 min_sales2_constr_105: Sold_1[105] = MIN ( total_stock_second_period_105
   , 1876.919481363728 )
 max_lost2_constr_105: Lost_1[105] = MAX ( Lost_1_expr_105 , 0 )
 GC636: Fraction[106] = LOGISTIC ( f_var[106] )
 min_sales_constr_106: Sold_0[106] = MIN ( Q0_var[106] , 297.5778032439039 )
 max_lost_constr_106: Lost_0[106] = MAX ( Lost_0_expr_106 , 0 )
 max_Q1_constr_106: Q1_var[106] = MAX ( Q_hat_adjusted_106 , 0 )
 min_sales2_constr_106: Sold_1[106] = MIN ( total_stock_second_period_106
   , 1523.209215594627 )
 max_lost2_constr_106: Lost_1[106] = MAX ( Lost_1_expr_106 , 0 )
 GC642: Fraction[107] = LOGISTIC ( f_var[107] )
 min_sales_constr_107: Sold_0[107] = MIN ( Q0_var[107] , 147.0119923328621 )
 max_lost_constr_107: Lost_0[107] = MAX ( Lost_0_expr_107 , 0 )
 max_Q1_constr_107: Q1_var[107] = MAX ( Q_hat_adjusted_107 , 0 )
 min_sales2_constr_107: Sold_1[107] = MIN ( total_stock_second_period_107
   , 642.4255277919862 )
 max_lost2_constr_107: Lost_1[107] = MAX ( Lost_1_expr_107 , 0 )
 GC648: Fraction[108] = LOGISTIC ( f_var[108] )
 min_sales_constr_108: Sold_0[108] = MIN ( Q0_var[108] , 295.8235141190507 )
 max_lost_constr_108: Lost_0[108] = MAX ( Lost_0_expr_108 , 0 )
 max_Q1_constr_108: Q1_var[108] = MAX ( Q_hat_adjusted_108 , 0 )
 min_sales2_constr_108: Sold_1[108] = MIN ( total_stock_second_period_108
   , 1256.63588680922 )
 max_lost2_constr_108: Lost_1[108] = MAX ( Lost_1_expr_108 , 0 )
 GC654: Fraction[109] = LOGISTIC ( f_var[109] )
 min_sales_constr_109: Sold_0[109] = MIN ( Q0_var[109] , 506.005740669968 )
 max_lost_constr_109: Lost_0[109] = MAX ( Lost_0_expr_109 , 0 )
 max_Q1_constr_109: Q1_var[109] = MAX ( Q_hat_adjusted_109 , 0 )
 min_sales2_constr_109: Sold_1[109] = MIN ( total_stock_second_period_109
   , 1317.311720029328 )
 max_lost2_constr_109: Lost_1[109] = MAX ( Lost_1_expr_109 , 0 )
 GC660: Fraction[110] = LOGISTIC ( f_var[110] )
 min_sales_constr_110: Sold_0[110] = MIN ( Q0_var[110] , 164.7969259052417 )
 max_lost_constr_110: Lost_0[110] = MAX ( Lost_0_expr_110 , 0 )
 max_Q1_constr_110: Q1_var[110] = MAX ( Q_hat_adjusted_110 , 0 )
 min_sales2_constr_110: Sold_1[110] = MIN ( total_stock_second_period_110
   , 593.7018169584919 )
 max_lost2_constr_110: Lost_1[110] = MAX ( Lost_1_expr_110 , 0 )
 GC666: Fraction[111] = LOGISTIC ( f_var[111] )
 min_sales_constr_111: Sold_0[111] = MIN ( Q0_var[111] , 347.4935862583258 )
 max_lost_constr_111: Lost_0[111] = MAX ( Lost_0_expr_111 , 0 )
 max_Q1_constr_111: Q1_var[111] = MAX ( Q_hat_adjusted_111 , 0 )
 min_sales2_constr_111: Sold_1[111] = MIN ( total_stock_second_period_111
   , 899.6992617283089 )
 max_lost2_constr_111: Lost_1[111] = MAX ( Lost_1_expr_111 , 0 )
 GC672: Fraction[112] = LOGISTIC ( f_var[112] )
 min_sales_constr_112: Sold_0[112] = MIN ( Q0_var[112] , 325.4657758465807 )
 max_lost_constr_112: Lost_0[112] = MAX ( Lost_0_expr_112 , 0 )
 max_Q1_constr_112: Q1_var[112] = MAX ( Q_hat_adjusted_112 , 0 )
 min_sales2_constr_112: Sold_1[112] = MIN ( total_stock_second_period_112
   , 1162.10442496777 )
 max_lost2_constr_112: Lost_1[112] = MAX ( Lost_1_expr_112 , 0 )
 GC678: Fraction[113] = LOGISTIC ( f_var[113] )
 min_sales_constr_113: Sold_0[113] = MIN ( Q0_var[113] , 226.4630016195126 )
 max_lost_constr_113: Lost_0[113] = MAX ( Lost_0_expr_113 , 0 )
 max_Q1_constr_113: Q1_var[113] = MAX ( Q_hat_adjusted_113 , 0 )
 min_sales2_constr_113: Sold_1[113] = MIN ( total_stock_second_period_113
   , 566.5965447154175 )
 max_lost2_constr_113: Lost_1[113] = MAX ( Lost_1_expr_113 , 0 )
 GC684: Fraction[114] = LOGISTIC ( f_var[114] )
 min_sales_constr_114: Sold_0[114] = MIN ( Q0_var[114] , 304.8590739172243 )
 max_lost_constr_114: Lost_0[114] = MAX ( Lost_0_expr_114 , 0 )
 max_Q1_constr_114: Q1_var[114] = MAX ( Q_hat_adjusted_114 , 0 )
 min_sales2_constr_114: Sold_1[114] = MIN ( total_stock_second_period_114
   , 1093.668741310533 )
 max_lost2_constr_114: Lost_1[114] = MAX ( Lost_1_expr_114 , 0 )
 GC690: Fraction[115] = LOGISTIC ( f_var[115] )
 min_sales_constr_115: Sold_0[115] = MIN ( Q0_var[115] , 194.0610952892391 )
 max_lost_constr_115: Lost_0[115] = MAX ( Lost_0_expr_115 , 0 )
 max_Q1_constr_115: Q1_var[115] = MAX ( Q_hat_adjusted_115 , 0 )
 min_sales2_constr_115: Sold_1[115] = MIN ( total_stock_second_period_115
   , 1023.130195168347 )
 max_lost2_constr_115: Lost_1[115] = MAX ( Lost_1_expr_115 , 0 )
 GC696: Fraction[116] = LOGISTIC ( f_var[116] )
 min_sales_constr_116: Sold_0[116] = MIN ( Q0_var[116] , 414.618201065166 )
 max_lost_constr_116: Lost_0[116] = MAX ( Lost_0_expr_116 , 0 )
 max_Q1_constr_116: Q1_var[116] = MAX ( Q_hat_adjusted_116 , 0 )
 min_sales2_constr_116: Sold_1[116] = MIN ( total_stock_second_period_116
   , 1143.534004850062 )
 max_lost2_constr_116: Lost_1[116] = MAX ( Lost_1_expr_116 , 0 )
 GC702: Fraction[117] = LOGISTIC ( f_var[117] )
 min_sales_constr_117: Sold_0[117] = MIN ( Q0_var[117] , 69.57894900517923 )
 max_lost_constr_117: Lost_0[117] = MAX ( Lost_0_expr_117 , 0 )
 max_Q1_constr_117: Q1_var[117] = MAX ( Q_hat_adjusted_117 , 0 )
 min_sales2_constr_117: Sold_1[117] = MIN ( total_stock_second_period_117
   , 460.3482183110156 )
 max_lost2_constr_117: Lost_1[117] = MAX ( Lost_1_expr_117 , 0 )
 GC708: Fraction[118] = LOGISTIC ( f_var[118] )
 min_sales_constr_118: Sold_0[118] = MIN ( Q0_var[118] , 379.8010388769554 )
 max_lost_constr_118: Lost_0[118] = MAX ( Lost_0_expr_118 , 0 )
 max_Q1_constr_118: Q1_var[118] = MAX ( Q_hat_adjusted_118 , 0 )
 min_sales2_constr_118: Sold_1[118] = MIN ( total_stock_second_period_118
   , 2493.894020546862 )
 max_lost2_constr_118: Lost_1[118] = MAX ( Lost_1_expr_118 , 0 )
 GC714: Fraction[119] = LOGISTIC ( f_var[119] )
 min_sales_constr_119: Sold_0[119] = MIN ( Q0_var[119] , 594.3387977609357 )
 max_lost_constr_119: Lost_0[119] = MAX ( Lost_0_expr_119 , 0 )
 max_Q1_constr_119: Q1_var[119] = MAX ( Q_hat_adjusted_119 , 0 )
 min_sales2_constr_119: Sold_1[119] = MIN ( total_stock_second_period_119
   , 1863.788427215063 )
 max_lost2_constr_119: Lost_1[119] = MAX ( Lost_1_expr_119 , 0 )
 GC720: Fraction[120] = LOGISTIC ( f_var[120] )
 min_sales_constr_120: Sold_0[120] = MIN ( Q0_var[120] , 281.2113416298586 )
 max_lost_constr_120: Lost_0[120] = MAX ( Lost_0_expr_120 , 0 )
 max_Q1_constr_120: Q1_var[120] = MAX ( Q_hat_adjusted_120 , 0 )
 min_sales2_constr_120: Sold_1[120] = MIN ( total_stock_second_period_120
   , 1494.35651349244 )
 max_lost2_constr_120: Lost_1[120] = MAX ( Lost_1_expr_120 , 0 )
 GC726: Fraction[121] = LOGISTIC ( f_var[121] )
 min_sales_constr_121: Sold_0[121] = MIN ( Q0_var[121] , 492.6667298467351 )
 max_lost_constr_121: Lost_0[121] = MAX ( Lost_0_expr_121 , 0 )
 max_Q1_constr_121: Q1_var[121] = MAX ( Q_hat_adjusted_121 , 0 )
 min_sales2_constr_121: Sold_1[121] = MIN ( total_stock_second_period_121
   , 2131.334263124195 )
 max_lost2_constr_121: Lost_1[121] = MAX ( Lost_1_expr_121 , 0 )
 GC732: Fraction[122] = LOGISTIC ( f_var[122] )
 min_sales_constr_122: Sold_0[122] = MIN ( Q0_var[122] , 441.9225754434032 )
 max_lost_constr_122: Lost_0[122] = MAX ( Lost_0_expr_122 , 0 )
 max_Q1_constr_122: Q1_var[122] = MAX ( Q_hat_adjusted_122 , 0 )
 min_sales2_constr_122: Sold_1[122] = MIN ( total_stock_second_period_122
   , 2150.220862066671 )
 max_lost2_constr_122: Lost_1[122] = MAX ( Lost_1_expr_122 , 0 )
 GC738: Fraction[123] = LOGISTIC ( f_var[123] )
 min_sales_constr_123: Sold_0[123] = MIN ( Q0_var[123] , 132.9698882223018 )
 max_lost_constr_123: Lost_0[123] = MAX ( Lost_0_expr_123 , 0 )
 max_Q1_constr_123: Q1_var[123] = MAX ( Q_hat_adjusted_123 , 0 )
 min_sales2_constr_123: Sold_1[123] = MIN ( total_stock_second_period_123
   , 874.5734396807923 )
 max_lost2_constr_123: Lost_1[123] = MAX ( Lost_1_expr_123 , 0 )
 GC744: Fraction[124] = LOGISTIC ( f_var[124] )
 min_sales_constr_124: Sold_0[124] = MIN ( Q0_var[124] , 237.7003103513032 )
 max_lost_constr_124: Lost_0[124] = MAX ( Lost_0_expr_124 , 0 )
 max_Q1_constr_124: Q1_var[124] = MAX ( Q_hat_adjusted_124 , 0 )
 min_sales2_constr_124: Sold_1[124] = MIN ( total_stock_second_period_124
   , 1781.278731126739 )
 max_lost2_constr_124: Lost_1[124] = MAX ( Lost_1_expr_124 , 0 )
 GC750: Fraction[125] = LOGISTIC ( f_var[125] )
 min_sales_constr_125: Sold_0[125] = MIN ( Q0_var[125] , 607.3807953967404 )
 max_lost_constr_125: Lost_0[125] = MAX ( Lost_0_expr_125 , 0 )
 max_Q1_constr_125: Q1_var[125] = MAX ( Q_hat_adjusted_125 , 0 )
 min_sales2_constr_125: Sold_1[125] = MIN ( total_stock_second_period_125
   , 2201.016465399305 )
 max_lost2_constr_125: Lost_1[125] = MAX ( Lost_1_expr_125 , 0 )
 GC756: Fraction[126] = LOGISTIC ( f_var[126] )
 min_sales_constr_126: Sold_0[126] = MIN ( Q0_var[126] , 612.072256724917 )
 max_lost_constr_126: Lost_0[126] = MAX ( Lost_0_expr_126 , 0 )
 max_Q1_constr_126: Q1_var[126] = MAX ( Q_hat_adjusted_126 , 0 )
 min_sales2_constr_126: Sold_1[126] = MIN ( total_stock_second_period_126
   , 2569.115550324982 )
 max_lost2_constr_126: Lost_1[126] = MAX ( Lost_1_expr_126 , 0 )
 GC762: Fraction[127] = LOGISTIC ( f_var[127] )
 min_sales_constr_127: Sold_0[127] = MIN ( Q0_var[127] , 499.9933882080895 )
 max_lost_constr_127: Lost_0[127] = MAX ( Lost_0_expr_127 , 0 )
 max_Q1_constr_127: Q1_var[127] = MAX ( Q_hat_adjusted_127 , 0 )
 min_sales2_constr_127: Sold_1[127] = MIN ( total_stock_second_period_127
   , 1146.24535678492 )
 max_lost2_constr_127: Lost_1[127] = MAX ( Lost_1_expr_127 , 0 )
 GC768: Fraction[128] = LOGISTIC ( f_var[128] )
 min_sales_constr_128: Sold_0[128] = MIN ( Q0_var[128] , 91.24704265603059 )
 max_lost_constr_128: Lost_0[128] = MAX ( Lost_0_expr_128 , 0 )
 max_Q1_constr_128: Q1_var[128] = MAX ( Q_hat_adjusted_128 , 0 )
 min_sales2_constr_128: Sold_1[128] = MIN ( total_stock_second_period_128
   , 616.4591849142245 )
 max_lost2_constr_128: Lost_1[128] = MAX ( Lost_1_expr_128 , 0 )
 GC774: Fraction[129] = LOGISTIC ( f_var[129] )
 min_sales_constr_129: Sold_0[129] = MIN ( Q0_var[129] , 343.9310519172101 )
 max_lost_constr_129: Lost_0[129] = MAX ( Lost_0_expr_129 , 0 )
 max_Q1_constr_129: Q1_var[129] = MAX ( Q_hat_adjusted_129 , 0 )
 min_sales2_constr_129: Sold_1[129] = MIN ( total_stock_second_period_129
   , 1565.309122799456 )
 max_lost2_constr_129: Lost_1[129] = MAX ( Lost_1_expr_129 , 0 )
 GC780: Fraction[130] = LOGISTIC ( f_var[130] )
 min_sales_constr_130: Sold_0[130] = MIN ( Q0_var[130] , 150.2829633311394 )
 max_lost_constr_130: Lost_0[130] = MAX ( Lost_0_expr_130 , 0 )
 max_Q1_constr_130: Q1_var[130] = MAX ( Q_hat_adjusted_130 , 0 )
 min_sales2_constr_130: Sold_1[130] = MIN ( total_stock_second_period_130
   , 430.9177596813311 )
 max_lost2_constr_130: Lost_1[130] = MAX ( Lost_1_expr_130 , 0 )
 GC786: Fraction[131] = LOGISTIC ( f_var[131] )
 min_sales_constr_131: Sold_0[131] = MIN ( Q0_var[131] , 128.1786537167152 )
 max_lost_constr_131: Lost_0[131] = MAX ( Lost_0_expr_131 , 0 )
 max_Q1_constr_131: Q1_var[131] = MAX ( Q_hat_adjusted_131 , 0 )
 min_sales2_constr_131: Sold_1[131] = MIN ( total_stock_second_period_131
   , 959.5639272525543 )
 max_lost2_constr_131: Lost_1[131] = MAX ( Lost_1_expr_131 , 0 )
 GC792: Fraction[132] = LOGISTIC ( f_var[132] )
 min_sales_constr_132: Sold_0[132] = MIN ( Q0_var[132] , 582.4663955143527 )
 max_lost_constr_132: Lost_0[132] = MAX ( Lost_0_expr_132 , 0 )
 max_Q1_constr_132: Q1_var[132] = MAX ( Q_hat_adjusted_132 , 0 )
 min_sales2_constr_132: Sold_1[132] = MIN ( total_stock_second_period_132
   , 2428.568912674506 )
 max_lost2_constr_132: Lost_1[132] = MAX ( Lost_1_expr_132 , 0 )
 GC798: Fraction[133] = LOGISTIC ( f_var[133] )
 min_sales_constr_133: Sold_0[133] = MIN ( Q0_var[133] , 169.0792382893863 )
 max_lost_constr_133: Lost_0[133] = MAX ( Lost_0_expr_133 , 0 )
 max_Q1_constr_133: Q1_var[133] = MAX ( Q_hat_adjusted_133 , 0 )
 min_sales2_constr_133: Sold_1[133] = MIN ( total_stock_second_period_133
   , 1208.332902807657 )
 max_lost2_constr_133: Lost_1[133] = MAX ( Lost_1_expr_133 , 0 )
 GC804: Fraction[134] = LOGISTIC ( f_var[134] )
 min_sales_constr_134: Sold_0[134] = MIN ( Q0_var[134] , 396.4351079840676 )
 max_lost_constr_134: Lost_0[134] = MAX ( Lost_0_expr_134 , 0 )
 max_Q1_constr_134: Q1_var[134] = MAX ( Q_hat_adjusted_134 , 0 )
 min_sales2_constr_134: Sold_1[134] = MIN ( total_stock_second_period_134
   , 1588.0688228323 )
 max_lost2_constr_134: Lost_1[134] = MAX ( Lost_1_expr_134 , 0 )
 GC810: Fraction[135] = LOGISTIC ( f_var[135] )
 min_sales_constr_135: Sold_0[135] = MIN ( Q0_var[135] , 431.1781818651522 )
 max_lost_constr_135: Lost_0[135] = MAX ( Lost_0_expr_135 , 0 )
 max_Q1_constr_135: Q1_var[135] = MAX ( Q_hat_adjusted_135 , 0 )
 min_sales2_constr_135: Sold_1[135] = MIN ( total_stock_second_period_135
   , 763.0687619328085 )
 max_lost2_constr_135: Lost_1[135] = MAX ( Lost_1_expr_135 , 0 )
 GC816: Fraction[136] = LOGISTIC ( f_var[136] )
 min_sales_constr_136: Sold_0[136] = MIN ( Q0_var[136] , 627.6920260953062 )
 max_lost_constr_136: Lost_0[136] = MAX ( Lost_0_expr_136 , 0 )
 max_Q1_constr_136: Q1_var[136] = MAX ( Q_hat_adjusted_136 , 0 )
 min_sales2_constr_136: Sold_1[136] = MIN ( total_stock_second_period_136
   , 2366.903109825657 )
 max_lost2_constr_136: Lost_1[136] = MAX ( Lost_1_expr_136 , 0 )
 GC822: Fraction[137] = LOGISTIC ( f_var[137] )
 min_sales_constr_137: Sold_0[137] = MIN ( Q0_var[137] , 409.6848311562389 )
 max_lost_constr_137: Lost_0[137] = MAX ( Lost_0_expr_137 , 0 )
 max_Q1_constr_137: Q1_var[137] = MAX ( Q_hat_adjusted_137 , 0 )
 min_sales2_constr_137: Sold_1[137] = MIN ( total_stock_second_period_137
   , 1910.576358528986 )
 max_lost2_constr_137: Lost_1[137] = MAX ( Lost_1_expr_137 , 0 )
 GC828: Fraction[138] = LOGISTIC ( f_var[138] )
 min_sales_constr_138: Sold_0[138] = MIN ( Q0_var[138] , 180.812043148223 )
 max_lost_constr_138: Lost_0[138] = MAX ( Lost_0_expr_138 , 0 )
 max_Q1_constr_138: Q1_var[138] = MAX ( Q_hat_adjusted_138 , 0 )
 min_sales2_constr_138: Sold_1[138] = MIN ( total_stock_second_period_138
   , 1034.750589107668 )
 max_lost2_constr_138: Lost_1[138] = MAX ( Lost_1_expr_138 , 0 )
 GC834: Fraction[139] = LOGISTIC ( f_var[139] )
 min_sales_constr_139: Sold_0[139] = MIN ( Q0_var[139] , 460.5228852169035 )
 max_lost_constr_139: Lost_0[139] = MAX ( Lost_0_expr_139 , 0 )
 max_Q1_constr_139: Q1_var[139] = MAX ( Q_hat_adjusted_139 , 0 )
 min_sales2_constr_139: Sold_1[139] = MIN ( total_stock_second_period_139
   , 2065.32762358913 )
 max_lost2_constr_139: Lost_1[139] = MAX ( Lost_1_expr_139 , 0 )
 GC840: Fraction[140] = LOGISTIC ( f_var[140] )
 min_sales_constr_140: Sold_0[140] = MIN ( Q0_var[140] , 311.7037429414744 )
 max_lost_constr_140: Lost_0[140] = MAX ( Lost_0_expr_140 , 0 )
 max_Q1_constr_140: Q1_var[140] = MAX ( Q_hat_adjusted_140 , 0 )
 min_sales2_constr_140: Sold_1[140] = MIN ( total_stock_second_period_140
   , 760.6952909757124 )
 max_lost2_constr_140: Lost_1[140] = MAX ( Lost_1_expr_140 , 0 )
 GC846: Fraction[141] = LOGISTIC ( f_var[141] )
 min_sales_constr_141: Sold_0[141] = MIN ( Q0_var[141] , 257.6424200299509 )
 max_lost_constr_141: Lost_0[141] = MAX ( Lost_0_expr_141 , 0 )
 max_Q1_constr_141: Q1_var[141] = MAX ( Q_hat_adjusted_141 , 0 )
 min_sales2_constr_141: Sold_1[141] = MIN ( total_stock_second_period_141
   , 1344.087351754543 )
 max_lost2_constr_141: Lost_1[141] = MAX ( Lost_1_expr_141 , 0 )
 GC852: Fraction[142] = LOGISTIC ( f_var[142] )
 min_sales_constr_142: Sold_0[142] = MIN ( Q0_var[142] , 158.7478957893452 )
 max_lost_constr_142: Lost_0[142] = MAX ( Lost_0_expr_142 , 0 )
 max_Q1_constr_142: Q1_var[142] = MAX ( Q_hat_adjusted_142 , 0 )
 min_sales2_constr_142: Sold_1[142] = MIN ( total_stock_second_period_142
   , 785.3959373659736 )
 max_lost2_constr_142: Lost_1[142] = MAX ( Lost_1_expr_142 , 0 )
 GC858: Fraction[143] = LOGISTIC ( f_var[143] )
 min_sales_constr_143: Sold_0[143] = MIN ( Q0_var[143] , 395.4554827517138 )
 max_lost_constr_143: Lost_0[143] = MAX ( Lost_0_expr_143 , 0 )
 max_Q1_constr_143: Q1_var[143] = MAX ( Q_hat_adjusted_143 , 0 )
 min_sales2_constr_143: Sold_1[143] = MIN ( total_stock_second_period_143
   , 2243.649026322146 )
 max_lost2_constr_143: Lost_1[143] = MAX ( Lost_1_expr_143 , 0 )
 GC864: Fraction[144] = LOGISTIC ( f_var[144] )
 min_sales_constr_144: Sold_0[144] = MIN ( Q0_var[144] , 88.29395789537652 )
 max_lost_constr_144: Lost_0[144] = MAX ( Lost_0_expr_144 , 0 )
 max_Q1_constr_144: Q1_var[144] = MAX ( Q_hat_adjusted_144 , 0 )
 min_sales2_constr_144: Sold_1[144] = MIN ( total_stock_second_period_144
   , 610.554720428368 )
 max_lost2_constr_144: Lost_1[144] = MAX ( Lost_1_expr_144 , 0 )
 GC870: Fraction[145] = LOGISTIC ( f_var[145] )
 min_sales_constr_145: Sold_0[145] = MIN ( Q0_var[145] , 297.1912583119029 )
 max_lost_constr_145: Lost_0[145] = MAX ( Lost_0_expr_145 , 0 )
 max_Q1_constr_145: Q1_var[145] = MAX ( Q_hat_adjusted_145 , 0 )
 min_sales2_constr_145: Sold_1[145] = MIN ( total_stock_second_period_145
   , 1123.575342018475 )
 max_lost2_constr_145: Lost_1[145] = MAX ( Lost_1_expr_145 , 0 )
 GC876: Fraction[146] = LOGISTIC ( f_var[146] )
 min_sales_constr_146: Sold_0[146] = MIN ( Q0_var[146] , 163.0980850681777 )
 max_lost_constr_146: Lost_0[146] = MAX ( Lost_0_expr_146 , 0 )
 max_Q1_constr_146: Q1_var[146] = MAX ( Q_hat_adjusted_146 , 0 )
 min_sales2_constr_146: Sold_1[146] = MIN ( total_stock_second_period_146
   , 1158.784798359883 )
 max_lost2_constr_146: Lost_1[146] = MAX ( Lost_1_expr_146 , 0 )
 GC882: Fraction[147] = LOGISTIC ( f_var[147] )
 min_sales_constr_147: Sold_0[147] = MIN ( Q0_var[147] , 259.4600328125892 )
 max_lost_constr_147: Lost_0[147] = MAX ( Lost_0_expr_147 , 0 )
 max_Q1_constr_147: Q1_var[147] = MAX ( Q_hat_adjusted_147 , 0 )
 min_sales2_constr_147: Sold_1[147] = MIN ( total_stock_second_period_147
   , 1468.671003214039 )
 max_lost2_constr_147: Lost_1[147] = MAX ( Lost_1_expr_147 , 0 )
 GC888: Fraction[148] = LOGISTIC ( f_var[148] )
 min_sales_constr_148: Sold_0[148] = MIN ( Q0_var[148] , 308.8668891196473 )
 max_lost_constr_148: Lost_0[148] = MAX ( Lost_0_expr_148 , 0 )
 max_Q1_constr_148: Q1_var[148] = MAX ( Q_hat_adjusted_148 , 0 )
 min_sales2_constr_148: Sold_1[148] = MIN ( total_stock_second_period_148
   , 1335.992750757796 )
 max_lost2_constr_148: Lost_1[148] = MAX ( Lost_1_expr_148 , 0 )
 GC894: Fraction[149] = LOGISTIC ( f_var[149] )
 min_sales_constr_149: Sold_0[149] = MIN ( Q0_var[149] , 491.9489027829728 )
 max_lost_constr_149: Lost_0[149] = MAX ( Lost_0_expr_149 , 0 )
 max_Q1_constr_149: Q1_var[149] = MAX ( Q_hat_adjusted_149 , 0 )
 min_sales2_constr_149: Sold_1[149] = MIN ( total_stock_second_period_149
   , 1970.735136213621 )
 max_lost2_constr_149: Lost_1[149] = MAX ( Lost_1_expr_149 , 0 )
 GC900: Fraction[150] = LOGISTIC ( f_var[150] )
 min_sales_constr_150: Sold_0[150] = MIN ( Q0_var[150] , 56.84210518658441 )
 max_lost_constr_150: Lost_0[150] = MAX ( Lost_0_expr_150 , 0 )
 max_Q1_constr_150: Q1_var[150] = MAX ( Q_hat_adjusted_150 , 0 )
 min_sales2_constr_150: Sold_1[150] = MIN ( total_stock_second_period_150
   , 444.8153705408223 )
 max_lost2_constr_150: Lost_1[150] = MAX ( Lost_1_expr_150 , 0 )
 GC906: Fraction[151] = LOGISTIC ( f_var[151] )
 min_sales_constr_151: Sold_0[151] = MIN ( Q0_var[151] , 593.3234035566596 )
 max_lost_constr_151: Lost_0[151] = MAX ( Lost_0_expr_151 , 0 )
 max_Q1_constr_151: Q1_var[151] = MAX ( Q_hat_adjusted_151 , 0 )
 min_sales2_constr_151: Sold_1[151] = MIN ( total_stock_second_period_151
   , 1616.26382576109 )
 max_lost2_constr_151: Lost_1[151] = MAX ( Lost_1_expr_151 , 0 )
 GC912: Fraction[152] = LOGISTIC ( f_var[152] )
 min_sales_constr_152: Sold_0[152] = MIN ( Q0_var[152] , 147.5218430662936 )
 max_lost_constr_152: Lost_0[152] = MAX ( Lost_0_expr_152 , 0 )
 max_Q1_constr_152: Q1_var[152] = MAX ( Q_hat_adjusted_152 , 0 )
 min_sales2_constr_152: Sold_1[152] = MIN ( total_stock_second_period_152
   , 1068.072225804329 )
 max_lost2_constr_152: Lost_1[152] = MAX ( Lost_1_expr_152 , 0 )
 GC918: Fraction[153] = LOGISTIC ( f_var[153] )
 min_sales_constr_153: Sold_0[153] = MIN ( Q0_var[153] , 438.273641190731 )
 max_lost_constr_153: Lost_0[153] = MAX ( Lost_0_expr_153 , 0 )
 max_Q1_constr_153: Q1_var[153] = MAX ( Q_hat_adjusted_153 , 0 )
 min_sales2_constr_153: Sold_1[153] = MIN ( total_stock_second_period_153
   , 907.8757321283105 )
 max_lost2_constr_153: Lost_1[153] = MAX ( Lost_1_expr_153 , 0 )
 GC924: Fraction[154] = LOGISTIC ( f_var[154] )
 min_sales_constr_154: Sold_0[154] = MIN ( Q0_var[154] , 341.4584729260217 )
 max_lost_constr_154: Lost_0[154] = MAX ( Lost_0_expr_154 , 0 )
 max_Q1_constr_154: Q1_var[154] = MAX ( Q_hat_adjusted_154 , 0 )
 min_sales2_constr_154: Sold_1[154] = MIN ( total_stock_second_period_154
   , 1949.136613191803 )
 max_lost2_constr_154: Lost_1[154] = MAX ( Lost_1_expr_154 , 0 )
 GC930: Fraction[155] = LOGISTIC ( f_var[155] )
 min_sales_constr_155: Sold_0[155] = MIN ( Q0_var[155] , 537.4245899083228 )
 max_lost_constr_155: Lost_0[155] = MAX ( Lost_0_expr_155 , 0 )
 max_Q1_constr_155: Q1_var[155] = MAX ( Q_hat_adjusted_155 , 0 )
 min_sales2_constr_155: Sold_1[155] = MIN ( total_stock_second_period_155
   , 2253.130409996704 )
 max_lost2_constr_155: Lost_1[155] = MAX ( Lost_1_expr_155 , 0 )
 GC936: Fraction[156] = LOGISTIC ( f_var[156] )
 min_sales_constr_156: Sold_0[156] = MIN ( Q0_var[156] , 202.4215749083674 )
 max_lost_constr_156: Lost_0[156] = MAX ( Lost_0_expr_156 , 0 )
 max_Q1_constr_156: Q1_var[156] = MAX ( Q_hat_adjusted_156 , 0 )
 min_sales2_constr_156: Sold_1[156] = MIN ( total_stock_second_period_156
   , 1396.343674834774 )
 max_lost2_constr_156: Lost_1[156] = MAX ( Lost_1_expr_156 , 0 )
 GC942: Fraction[157] = LOGISTIC ( f_var[157] )
 min_sales_constr_157: Sold_0[157] = MIN ( Q0_var[157] , 698.8419203542333 )
 max_lost_constr_157: Lost_0[157] = MAX ( Lost_0_expr_157 , 0 )
 max_Q1_constr_157: Q1_var[157] = MAX ( Q_hat_adjusted_157 , 0 )
 min_sales2_constr_157: Sold_1[157] = MIN ( total_stock_second_period_157
   , 2051.825627528687 )
 max_lost2_constr_157: Lost_1[157] = MAX ( Lost_1_expr_157 , 0 )
 GC948: Fraction[158] = LOGISTIC ( f_var[158] )
 min_sales_constr_158: Sold_0[158] = MIN ( Q0_var[158] , 75.97277971462395 )
 max_lost_constr_158: Lost_0[158] = MAX ( Lost_0_expr_158 , 0 )
 max_Q1_constr_158: Q1_var[158] = MAX ( Q_hat_adjusted_158 , 0 )
 min_sales2_constr_158: Sold_1[158] = MIN ( total_stock_second_period_158
   , 585.3635202546136 )
 max_lost2_constr_158: Lost_1[158] = MAX ( Lost_1_expr_158 , 0 )
 GC954: Fraction[159] = LOGISTIC ( f_var[159] )
 min_sales_constr_159: Sold_0[159] = MIN ( Q0_var[159] , 524.5361651542147 )
 max_lost_constr_159: Lost_0[159] = MAX ( Lost_0_expr_159 , 0 )
 max_Q1_constr_159: Q1_var[159] = MAX ( Q_hat_adjusted_159 , 0 )
 min_sales2_constr_159: Sold_1[159] = MIN ( total_stock_second_period_159
   , 1528.725534777989 )
 max_lost2_constr_159: Lost_1[159] = MAX ( Lost_1_expr_159 , 0 )
 GC960: Fraction[160] = LOGISTIC ( f_var[160] )
 min_sales_constr_160: Sold_0[160] = MIN ( Q0_var[160] , 444.0140374725378 )
 max_lost_constr_160: Lost_0[160] = MAX ( Lost_0_expr_160 , 0 )
 max_Q1_constr_160: Q1_var[160] = MAX ( Q_hat_adjusted_160 , 0 )
 min_sales2_constr_160: Sold_1[160] = MIN ( total_stock_second_period_160
   , 2133.806755197715 )
 max_lost2_constr_160: Lost_1[160] = MAX ( Lost_1_expr_160 , 0 )
 GC966: Fraction[161] = LOGISTIC ( f_var[161] )
 min_sales_constr_161: Sold_0[161] = MIN ( Q0_var[161] , 239.3424487672534 )
 max_lost_constr_161: Lost_0[161] = MAX ( Lost_0_expr_161 , 0 )
 max_Q1_constr_161: Q1_var[161] = MAX ( Q_hat_adjusted_161 , 0 )
 min_sales2_constr_161: Sold_1[161] = MIN ( total_stock_second_period_161
   , 865.4934553755439 )
 max_lost2_constr_161: Lost_1[161] = MAX ( Lost_1_expr_161 , 0 )
 GC972: Fraction[162] = LOGISTIC ( f_var[162] )
 min_sales_constr_162: Sold_0[162] = MIN ( Q0_var[162] , 403.676960995897 )
 max_lost_constr_162: Lost_0[162] = MAX ( Lost_0_expr_162 , 0 )
 max_Q1_constr_162: Q1_var[162] = MAX ( Q_hat_adjusted_162 , 0 )
 min_sales2_constr_162: Sold_1[162] = MIN ( total_stock_second_period_162
   , 1421.91640672116 )
 max_lost2_constr_162: Lost_1[162] = MAX ( Lost_1_expr_162 , 0 )
 GC978: Fraction[163] = LOGISTIC ( f_var[163] )
 min_sales_constr_163: Sold_0[163] = MIN ( Q0_var[163] , 499.8746860815196 )
 max_lost_constr_163: Lost_0[163] = MAX ( Lost_0_expr_163 , 0 )
 max_Q1_constr_163: Q1_var[163] = MAX ( Q_hat_adjusted_163 , 0 )
 min_sales2_constr_163: Sold_1[163] = MIN ( total_stock_second_period_163
   , 2180.088476356245 )
 max_lost2_constr_163: Lost_1[163] = MAX ( Lost_1_expr_163 , 0 )
 GC984: Fraction[164] = LOGISTIC ( f_var[164] )
 min_sales_constr_164: Sold_0[164] = MIN ( Q0_var[164] , 178.869011957668 )
 max_lost_constr_164: Lost_0[164] = MAX ( Lost_0_expr_164 , 0 )
 max_Q1_constr_164: Q1_var[164] = MAX ( Q_hat_adjusted_164 , 0 )
 min_sales2_constr_164: Sold_1[164] = MIN ( total_stock_second_period_164
   , 491.3409414359294 )
 max_lost2_constr_164: Lost_1[164] = MAX ( Lost_1_expr_164 , 0 )
 GC990: Fraction[165] = LOGISTIC ( f_var[165] )
 min_sales_constr_165: Sold_0[165] = MIN ( Q0_var[165] , 120.0639925704784 )
 max_lost_constr_165: Lost_0[165] = MAX ( Lost_0_expr_165 , 0 )
 max_Q1_constr_165: Q1_var[165] = MAX ( Q_hat_adjusted_165 , 0 )
 min_sales2_constr_165: Sold_1[165] = MIN ( total_stock_second_period_165
   , 809.0733215218597 )
 max_lost2_constr_165: Lost_1[165] = MAX ( Lost_1_expr_165 , 0 )
 GC996: Fraction[166] = LOGISTIC ( f_var[166] )
 min_sales_constr_166: Sold_0[166] = MIN ( Q0_var[166] , 445.6641819954741 )
 max_lost_constr_166: Lost_0[166] = MAX ( Lost_0_expr_166 , 0 )
 max_Q1_constr_166: Q1_var[166] = MAX ( Q_hat_adjusted_166 , 0 )
 min_sales2_constr_166: Sold_1[166] = MIN ( total_stock_second_period_166
   , 1745.544693587992 )
 max_lost2_constr_166: Lost_1[166] = MAX ( Lost_1_expr_166 , 0 )
 GC1002: Fraction[167] = LOGISTIC ( f_var[167] )
 min_sales_constr_167: Sold_0[167] = MIN ( Q0_var[167] , 531.4675974688647 )
 max_lost_constr_167: Lost_0[167] = MAX ( Lost_0_expr_167 , 0 )
 max_Q1_constr_167: Q1_var[167] = MAX ( Q_hat_adjusted_167 , 0 )
 min_sales2_constr_167: Sold_1[167] = MIN ( total_stock_second_period_167
   , 1323.871842602416 )
 max_lost2_constr_167: Lost_1[167] = MAX ( Lost_1_expr_167 , 0 )
 GC1008: Fraction[168] = LOGISTIC ( f_var[168] )
 min_sales_constr_168: Sold_0[168] = MIN ( Q0_var[168] , 565.2566982574215 )
 max_lost_constr_168: Lost_0[168] = MAX ( Lost_0_expr_168 , 0 )
 max_Q1_constr_168: Q1_var[168] = MAX ( Q_hat_adjusted_168 , 0 )
 min_sales2_constr_168: Sold_1[168] = MIN ( total_stock_second_period_168
   , 1953.105752867165 )
 max_lost2_constr_168: Lost_1[168] = MAX ( Lost_1_expr_168 , 0 )
 GC1014: Fraction[169] = LOGISTIC ( f_var[169] )
 min_sales_constr_169: Sold_0[169] = MIN ( Q0_var[169] , 610.4619027649469 )
 max_lost_constr_169: Lost_0[169] = MAX ( Lost_0_expr_169 , 0 )
 max_Q1_constr_169: Q1_var[169] = MAX ( Q_hat_adjusted_169 , 0 )
 min_sales2_constr_169: Sold_1[169] = MIN ( total_stock_second_period_169
   , 1717.135387924426 )
 max_lost2_constr_169: Lost_1[169] = MAX ( Lost_1_expr_169 , 0 )
 GC1020: Fraction[170] = LOGISTIC ( f_var[170] )
 min_sales_constr_170: Sold_0[170] = MIN ( Q0_var[170] , 264.7188384078374 )
 max_lost_constr_170: Lost_0[170] = MAX ( Lost_0_expr_170 , 0 )
 max_Q1_constr_170: Q1_var[170] = MAX ( Q_hat_adjusted_170 , 0 )
 min_sales2_constr_170: Sold_1[170] = MIN ( total_stock_second_period_170
   , 867.3899397526766 )
 max_lost2_constr_170: Lost_1[170] = MAX ( Lost_1_expr_170 , 0 )
 GC1026: Fraction[171] = LOGISTIC ( f_var[171] )
 min_sales_constr_171: Sold_0[171] = MIN ( Q0_var[171] , 275.8069251597127 )
 max_lost_constr_171: Lost_0[171] = MAX ( Lost_0_expr_171 , 0 )
 max_Q1_constr_171: Q1_var[171] = MAX ( Q_hat_adjusted_171 , 0 )
 min_sales2_constr_171: Sold_1[171] = MIN ( total_stock_second_period_171
   , 1440.908027352427 )
 max_lost2_constr_171: Lost_1[171] = MAX ( Lost_1_expr_171 , 0 )
 GC1032: Fraction[172] = LOGISTIC ( f_var[172] )
 min_sales_constr_172: Sold_0[172] = MIN ( Q0_var[172] , 245.5623510792364 )
 max_lost_constr_172: Lost_0[172] = MAX ( Lost_0_expr_172 , 0 )
 max_Q1_constr_172: Q1_var[172] = MAX ( Q_hat_adjusted_172 , 0 )
 min_sales2_constr_172: Sold_1[172] = MIN ( total_stock_second_period_172
   , 1222.357150859487 )
 max_lost2_constr_172: Lost_1[172] = MAX ( Lost_1_expr_172 , 0 )
 GC1038: Fraction[173] = LOGISTIC ( f_var[173] )
 min_sales_constr_173: Sold_0[173] = MIN ( Q0_var[173] , 400.02188960258 )
 max_lost_constr_173: Lost_0[173] = MAX ( Lost_0_expr_173 , 0 )
 max_Q1_constr_173: Q1_var[173] = MAX ( Q_hat_adjusted_173 , 0 )
 min_sales2_constr_173: Sold_1[173] = MIN ( total_stock_second_period_173
   , 1181.695170898573 )
 max_lost2_constr_173: Lost_1[173] = MAX ( Lost_1_expr_173 , 0 )
 GC1044: Fraction[174] = LOGISTIC ( f_var[174] )
 min_sales_constr_174: Sold_0[174] = MIN ( Q0_var[174] , 574.7071782693196 )
 max_lost_constr_174: Lost_0[174] = MAX ( Lost_0_expr_174 , 0 )
 max_Q1_constr_174: Q1_var[174] = MAX ( Q_hat_adjusted_174 , 0 )
 min_sales2_constr_174: Sold_1[174] = MIN ( total_stock_second_period_174
   , 2186.666447987837 )
 max_lost2_constr_174: Lost_1[174] = MAX ( Lost_1_expr_174 , 0 )
 GC1050: Fraction[175] = LOGISTIC ( f_var[175] )
 min_sales_constr_175: Sold_0[175] = MIN ( Q0_var[175] , 427.174132175296 )
 max_lost_constr_175: Lost_0[175] = MAX ( Lost_0_expr_175 , 0 )
 max_Q1_constr_175: Q1_var[175] = MAX ( Q_hat_adjusted_175 , 0 )
 min_sales2_constr_175: Sold_1[175] = MIN ( total_stock_second_period_175
   , 1494.365643430667 )
 max_lost2_constr_175: Lost_1[175] = MAX ( Lost_1_expr_175 , 0 )
 GC1056: Fraction[176] = LOGISTIC ( f_var[176] )
 min_sales_constr_176: Sold_0[176] = MIN ( Q0_var[176] , 355.7967346945577 )
 max_lost_constr_176: Lost_0[176] = MAX ( Lost_0_expr_176 , 0 )
 max_Q1_constr_176: Q1_var[176] = MAX ( Q_hat_adjusted_176 , 0 )
 min_sales2_constr_176: Sold_1[176] = MIN ( total_stock_second_period_176
   , 1419.34095129156 )
 max_lost2_constr_176: Lost_1[176] = MAX ( Lost_1_expr_176 , 0 )
 GC1062: Fraction[177] = LOGISTIC ( f_var[177] )
 min_sales_constr_177: Sold_0[177] = MIN ( Q0_var[177] , 213.996161354379 )
 max_lost_constr_177: Lost_0[177] = MAX ( Lost_0_expr_177 , 0 )
 max_Q1_constr_177: Q1_var[177] = MAX ( Q_hat_adjusted_177 , 0 )
 min_sales2_constr_177: Sold_1[177] = MIN ( total_stock_second_period_177
   , 1071.755748224121 )
 max_lost2_constr_177: Lost_1[177] = MAX ( Lost_1_expr_177 , 0 )
 GC1068: Fraction[178] = LOGISTIC ( f_var[178] )
 min_sales_constr_178: Sold_0[178] = MIN ( Q0_var[178] , 695.6388548663745 )
 max_lost_constr_178: Lost_0[178] = MAX ( Lost_0_expr_178 , 0 )
 max_Q1_constr_178: Q1_var[178] = MAX ( Q_hat_adjusted_178 , 0 )
 min_sales2_constr_178: Sold_1[178] = MIN ( total_stock_second_period_178
   , 2372.978510954937 )
 max_lost2_constr_178: Lost_1[178] = MAX ( Lost_1_expr_178 , 0 )
 GC1074: Fraction[179] = LOGISTIC ( f_var[179] )
 min_sales_constr_179: Sold_0[179] = MIN ( Q0_var[179] , 274.0257614515539 )
 max_lost_constr_179: Lost_0[179] = MAX ( Lost_0_expr_179 , 0 )
 max_Q1_constr_179: Q1_var[179] = MAX ( Q_hat_adjusted_179 , 0 )
 min_sales2_constr_179: Sold_1[179] = MIN ( total_stock_second_period_179
   , 843.5877060852167 )
 max_lost2_constr_179: Lost_1[179] = MAX ( Lost_1_expr_179 , 0 )
 GC1080: Fraction[180] = LOGISTIC ( f_var[180] )
 min_sales_constr_180: Sold_0[180] = MIN ( Q0_var[180] , 329.2425263950671 )
 max_lost_constr_180: Lost_0[180] = MAX ( Lost_0_expr_180 , 0 )
 max_Q1_constr_180: Q1_var[180] = MAX ( Q_hat_adjusted_180 , 0 )
 min_sales2_constr_180: Sold_1[180] = MIN ( total_stock_second_period_180
   , 1800.426786142918 )
 max_lost2_constr_180: Lost_1[180] = MAX ( Lost_1_expr_180 , 0 )
 GC1086: Fraction[181] = LOGISTIC ( f_var[181] )
 min_sales_constr_181: Sold_0[181] = MIN ( Q0_var[181] , 578.6144911696799 )
 max_lost_constr_181: Lost_0[181] = MAX ( Lost_0_expr_181 , 0 )
 max_Q1_constr_181: Q1_var[181] = MAX ( Q_hat_adjusted_181 , 0 )
 min_sales2_constr_181: Sold_1[181] = MIN ( total_stock_second_period_181
   , 2038.985575954113 )
 max_lost2_constr_181: Lost_1[181] = MAX ( Lost_1_expr_181 , 0 )
 GC1092: Fraction[182] = LOGISTIC ( f_var[182] )
 min_sales_constr_182: Sold_0[182] = MIN ( Q0_var[182] , 407.1374655694316 )
 max_lost_constr_182: Lost_0[182] = MAX ( Lost_0_expr_182 , 0 )
 max_Q1_constr_182: Q1_var[182] = MAX ( Q_hat_adjusted_182 , 0 )
 min_sales2_constr_182: Sold_1[182] = MIN ( total_stock_second_period_182
   , 1614.874399577545 )
 max_lost2_constr_182: Lost_1[182] = MAX ( Lost_1_expr_182 , 0 )
 GC1098: Fraction[183] = LOGISTIC ( f_var[183] )
 min_sales_constr_183: Sold_0[183] = MIN ( Q0_var[183] , 289.1019003866597 )
 max_lost_constr_183: Lost_0[183] = MAX ( Lost_0_expr_183 , 0 )
 max_Q1_constr_183: Q1_var[183] = MAX ( Q_hat_adjusted_183 , 0 )
 min_sales2_constr_183: Sold_1[183] = MIN ( total_stock_second_period_183
   , 1371.20119352674 )
 max_lost2_constr_183: Lost_1[183] = MAX ( Lost_1_expr_183 , 0 )
 GC1104: Fraction[184] = LOGISTIC ( f_var[184] )
 min_sales_constr_184: Sold_0[184] = MIN ( Q0_var[184] , 208.8372286312708 )
 max_lost_constr_184: Lost_0[184] = MAX ( Lost_0_expr_184 , 0 )
 max_Q1_constr_184: Q1_var[184] = MAX ( Q_hat_adjusted_184 , 0 )
 min_sales2_constr_184: Sold_1[184] = MIN ( total_stock_second_period_184
   , 992.1224068965528 )
 max_lost2_constr_184: Lost_1[184] = MAX ( Lost_1_expr_184 , 0 )
 GC1110: Fraction[185] = LOGISTIC ( f_var[185] )
 min_sales_constr_185: Sold_0[185] = MIN ( Q0_var[185] , 243.4325675892941 )
 max_lost_constr_185: Lost_0[185] = MAX ( Lost_0_expr_185 , 0 )
 max_Q1_constr_185: Q1_var[185] = MAX ( Q_hat_adjusted_185 , 0 )
 min_sales2_constr_185: Sold_1[185] = MIN ( total_stock_second_period_185
   , 1122.379430706891 )
 max_lost2_constr_185: Lost_1[185] = MAX ( Lost_1_expr_185 , 0 )
 GC1116: Fraction[186] = LOGISTIC ( f_var[186] )
 min_sales_constr_186: Sold_0[186] = MIN ( Q0_var[186] , 523.8994973547306 )
 max_lost_constr_186: Lost_0[186] = MAX ( Lost_0_expr_186 , 0 )
 max_Q1_constr_186: Q1_var[186] = MAX ( Q_hat_adjusted_186 , 0 )
 min_sales2_constr_186: Sold_1[186] = MIN ( total_stock_second_period_186
   , 1822.958237429521 )
 max_lost2_constr_186: Lost_1[186] = MAX ( Lost_1_expr_186 , 0 )
 GC1122: Fraction[187] = LOGISTIC ( f_var[187] )
 min_sales_constr_187: Sold_0[187] = MIN ( Q0_var[187] , 383.8936940510458 )
 max_lost_constr_187: Lost_0[187] = MAX ( Lost_0_expr_187 , 0 )
 max_Q1_constr_187: Q1_var[187] = MAX ( Q_hat_adjusted_187 , 0 )
 min_sales2_constr_187: Sold_1[187] = MIN ( total_stock_second_period_187
   , 1826.742474944972 )
 max_lost2_constr_187: Lost_1[187] = MAX ( Lost_1_expr_187 , 0 )
 GC1128: Fraction[188] = LOGISTIC ( f_var[188] )
 min_sales_constr_188: Sold_0[188] = MIN ( Q0_var[188] , 498.6107403156558 )
 max_lost_constr_188: Lost_0[188] = MAX ( Lost_0_expr_188 , 0 )
 max_Q1_constr_188: Q1_var[188] = MAX ( Q_hat_adjusted_188 , 0 )
 min_sales2_constr_188: Sold_1[188] = MIN ( total_stock_second_period_188
   , 2360.409567194891 )
 max_lost2_constr_188: Lost_1[188] = MAX ( Lost_1_expr_188 , 0 )
 GC1134: Fraction[189] = LOGISTIC ( f_var[189] )
 min_sales_constr_189: Sold_0[189] = MIN ( Q0_var[189] , 742.0075371204716 )
 max_lost_constr_189: Lost_0[189] = MAX ( Lost_0_expr_189 , 0 )
 max_Q1_constr_189: Q1_var[189] = MAX ( Q_hat_adjusted_189 , 0 )
 min_sales2_constr_189: Sold_1[189] = MIN ( total_stock_second_period_189
   , 1889.239071403888 )
 max_lost2_constr_189: Lost_1[189] = MAX ( Lost_1_expr_189 , 0 )
 GC1140: Fraction[190] = LOGISTIC ( f_var[190] )
 min_sales_constr_190: Sold_0[190] = MIN ( Q0_var[190] , 493.9450427889806 )
 max_lost_constr_190: Lost_0[190] = MAX ( Lost_0_expr_190 , 0 )
 max_Q1_constr_190: Q1_var[190] = MAX ( Q_hat_adjusted_190 , 0 )
 min_sales2_constr_190: Sold_1[190] = MIN ( total_stock_second_period_190
   , 2029.368961379122 )
 max_lost2_constr_190: Lost_1[190] = MAX ( Lost_1_expr_190 , 0 )
 GC1146: Fraction[191] = LOGISTIC ( f_var[191] )
 min_sales_constr_191: Sold_0[191] = MIN ( Q0_var[191] , 432.8494975223789 )
 max_lost_constr_191: Lost_0[191] = MAX ( Lost_0_expr_191 , 0 )
 max_Q1_constr_191: Q1_var[191] = MAX ( Q_hat_adjusted_191 , 0 )
 min_sales2_constr_191: Sold_1[191] = MIN ( total_stock_second_period_191
   , 2423.534447115342 )
 max_lost2_constr_191: Lost_1[191] = MAX ( Lost_1_expr_191 , 0 )
 GC1152: Fraction[192] = LOGISTIC ( f_var[192] )
 min_sales_constr_192: Sold_0[192] = MIN ( Q0_var[192] , 119.0798743351343 )
 max_lost_constr_192: Lost_0[192] = MAX ( Lost_0_expr_192 , 0 )
 max_Q1_constr_192: Q1_var[192] = MAX ( Q_hat_adjusted_192 , 0 )
 min_sales2_constr_192: Sold_1[192] = MIN ( total_stock_second_period_192
   , 1250.971345856744 )
 max_lost2_constr_192: Lost_1[192] = MAX ( Lost_1_expr_192 , 0 )
 GC1158: Fraction[193] = LOGISTIC ( f_var[193] )
 min_sales_constr_193: Sold_0[193] = MIN ( Q0_var[193] , 264.6647972575785 )
 max_lost_constr_193: Lost_0[193] = MAX ( Lost_0_expr_193 , 0 )
 max_Q1_constr_193: Q1_var[193] = MAX ( Q_hat_adjusted_193 , 0 )
 min_sales2_constr_193: Sold_1[193] = MIN ( total_stock_second_period_193
   , 1311.615246472861 )
 max_lost2_constr_193: Lost_1[193] = MAX ( Lost_1_expr_193 , 0 )
 GC1164: Fraction[194] = LOGISTIC ( f_var[194] )
 min_sales_constr_194: Sold_0[194] = MIN ( Q0_var[194] , 230.2915183824858 )
 max_lost_constr_194: Lost_0[194] = MAX ( Lost_0_expr_194 , 0 )
 max_Q1_constr_194: Q1_var[194] = MAX ( Q_hat_adjusted_194 , 0 )
 min_sales2_constr_194: Sold_1[194] = MIN ( total_stock_second_period_194
   , 1066.820532836279 )
 max_lost2_constr_194: Lost_1[194] = MAX ( Lost_1_expr_194 , 0 )
 GC1170: Fraction[195] = LOGISTIC ( f_var[195] )
 min_sales_constr_195: Sold_0[195] = MIN ( Q0_var[195] , 528.0286649174883 )
 max_lost_constr_195: Lost_0[195] = MAX ( Lost_0_expr_195 , 0 )
 max_Q1_constr_195: Q1_var[195] = MAX ( Q_hat_adjusted_195 , 0 )
 min_sales2_constr_195: Sold_1[195] = MIN ( total_stock_second_period_195
   , 1783.145862078497 )
 max_lost2_constr_195: Lost_1[195] = MAX ( Lost_1_expr_195 , 0 )
 GC1176: Fraction[196] = LOGISTIC ( f_var[196] )
 min_sales_constr_196: Sold_0[196] = MIN ( Q0_var[196] , 545.0957351606476 )
 max_lost_constr_196: Lost_0[196] = MAX ( Lost_0_expr_196 , 0 )
 max_Q1_constr_196: Q1_var[196] = MAX ( Q_hat_adjusted_196 , 0 )
 min_sales2_constr_196: Sold_1[196] = MIN ( total_stock_second_period_196
   , 1555.276832110032 )
 max_lost2_constr_196: Lost_1[196] = MAX ( Lost_1_expr_196 , 0 )
 GC1182: Fraction[197] = LOGISTIC ( f_var[197] )
 min_sales_constr_197: Sold_0[197] = MIN ( Q0_var[197] , 128.2293070037987 )
 max_lost_constr_197: Lost_0[197] = MAX ( Lost_0_expr_197 , 0 )
 max_Q1_constr_197: Q1_var[197] = MAX ( Q_hat_adjusted_197 , 0 )
 min_sales2_constr_197: Sold_1[197] = MIN ( total_stock_second_period_197
   , 1160.458602694817 )
 max_lost2_constr_197: Lost_1[197] = MAX ( Lost_1_expr_197 , 0 )
 GC1188: Fraction[198] = LOGISTIC ( f_var[198] )
 min_sales_constr_198: Sold_0[198] = MIN ( Q0_var[198] , 147.5668046372598 )
 max_lost_constr_198: Lost_0[198] = MAX ( Lost_0_expr_198 , 0 )
 max_Q1_constr_198: Q1_var[198] = MAX ( Q_hat_adjusted_198 , 0 )
 min_sales2_constr_198: Sold_1[198] = MIN ( total_stock_second_period_198
   , 451.8371744690056 )
 max_lost2_constr_198: Lost_1[198] = MAX ( Lost_1_expr_198 , 0 )
 GC1194: Fraction[199] = LOGISTIC ( f_var[199] )
 min_sales_constr_199: Sold_0[199] = MIN ( Q0_var[199] , 149.8046591276339 )
 max_lost_constr_199: Lost_0[199] = MAX ( Lost_0_expr_199 , 0 )
 max_Q1_constr_199: Q1_var[199] = MAX ( Q_hat_adjusted_199 , 0 )
 min_sales2_constr_199: Sold_1[199] = MIN ( total_stock_second_period_199
   , 1188.538643362749 )
 max_lost2_constr_199: Lost_1[199] = MAX ( Lost_1_expr_199 , 0 )
 GC1200: Fraction[200] = LOGISTIC ( f_var[200] )
 min_sales_constr_200: Sold_0[200] = MIN ( Q0_var[200] , 294.7966953176393 )
 max_lost_constr_200: Lost_0[200] = MAX ( Lost_0_expr_200 , 0 )
 max_Q1_constr_200: Q1_var[200] = MAX ( Q_hat_adjusted_200 , 0 )
 min_sales2_constr_200: Sold_1[200] = MIN ( total_stock_second_period_200
   , 1052.020125747054 )
 max_lost2_constr_200: Lost_1[200] = MAX ( Lost_1_expr_200 , 0 )
 GC1206: Fraction[201] = LOGISTIC ( f_var[201] )
 min_sales_constr_201: Sold_0[201] = MIN ( Q0_var[201] , 499.3142297197068 )
 max_lost_constr_201: Lost_0[201] = MAX ( Lost_0_expr_201 , 0 )
 max_Q1_constr_201: Q1_var[201] = MAX ( Q_hat_adjusted_201 , 0 )
 min_sales2_constr_201: Sold_1[201] = MIN ( total_stock_second_period_201
   , 2317.277212541023 )
 max_lost2_constr_201: Lost_1[201] = MAX ( Lost_1_expr_201 , 0 )
 GC1212: Fraction[202] = LOGISTIC ( f_var[202] )
 min_sales_constr_202: Sold_0[202] = MIN ( Q0_var[202] , 728.4740636010729 )
 max_lost_constr_202: Lost_0[202] = MAX ( Lost_0_expr_202 , 0 )
 max_Q1_constr_202: Q1_var[202] = MAX ( Q_hat_adjusted_202 , 0 )
 min_sales2_constr_202: Sold_1[202] = MIN ( total_stock_second_period_202
   , 2008.803261169839 )
 max_lost2_constr_202: Lost_1[202] = MAX ( Lost_1_expr_202 , 0 )
 GC1218: Fraction[203] = LOGISTIC ( f_var[203] )
 min_sales_constr_203: Sold_0[203] = MIN ( Q0_var[203] , 544.6209260811635 )
 max_lost_constr_203: Lost_0[203] = MAX ( Lost_0_expr_203 , 0 )
 max_Q1_constr_203: Q1_var[203] = MAX ( Q_hat_adjusted_203 , 0 )
 min_sales2_constr_203: Sold_1[203] = MIN ( total_stock_second_period_203
   , 2126.074526386978 )
 max_lost2_constr_203: Lost_1[203] = MAX ( Lost_1_expr_203 , 0 )
 GC1224: Fraction[204] = LOGISTIC ( f_var[204] )
 min_sales_constr_204: Sold_0[204] = MIN ( Q0_var[204] , 562.5581607230132 )
 max_lost_constr_204: Lost_0[204] = MAX ( Lost_0_expr_204 , 0 )
 max_Q1_constr_204: Q1_var[204] = MAX ( Q_hat_adjusted_204 , 0 )
 min_sales2_constr_204: Sold_1[204] = MIN ( total_stock_second_period_204
   , 1671.966694022811 )
 max_lost2_constr_204: Lost_1[204] = MAX ( Lost_1_expr_204 , 0 )
 GC1230: Fraction[205] = LOGISTIC ( f_var[205] )
 min_sales_constr_205: Sold_0[205] = MIN ( Q0_var[205] , 484.0102663735216 )
 max_lost_constr_205: Lost_0[205] = MAX ( Lost_0_expr_205 , 0 )
 max_Q1_constr_205: Q1_var[205] = MAX ( Q_hat_adjusted_205 , 0 )
 min_sales2_constr_205: Sold_1[205] = MIN ( total_stock_second_period_205
   , 1027.965803012694 )
 max_lost2_constr_205: Lost_1[205] = MAX ( Lost_1_expr_205 , 0 )
 GC1236: Fraction[206] = LOGISTIC ( f_var[206] )
 min_sales_constr_206: Sold_0[206] = MIN ( Q0_var[206] , 187.2122906779878 )
 max_lost_constr_206: Lost_0[206] = MAX ( Lost_0_expr_206 , 0 )
 max_Q1_constr_206: Q1_var[206] = MAX ( Q_hat_adjusted_206 , 0 )
 min_sales2_constr_206: Sold_1[206] = MIN ( total_stock_second_period_206
   , 1689.26904540914 )
 max_lost2_constr_206: Lost_1[206] = MAX ( Lost_1_expr_206 , 0 )
 GC1242: Fraction[207] = LOGISTIC ( f_var[207] )
 min_sales_constr_207: Sold_0[207] = MIN ( Q0_var[207] , 385.6634294300526 )
 max_lost_constr_207: Lost_0[207] = MAX ( Lost_0_expr_207 , 0 )
 max_Q1_constr_207: Q1_var[207] = MAX ( Q_hat_adjusted_207 , 0 )
 min_sales2_constr_207: Sold_1[207] = MIN ( total_stock_second_period_207
   , 2063.274228670299 )
 max_lost2_constr_207: Lost_1[207] = MAX ( Lost_1_expr_207 , 0 )
 GC1248: Fraction[208] = LOGISTIC ( f_var[208] )
 min_sales_constr_208: Sold_0[208] = MIN ( Q0_var[208] , 175.2189204002573 )
 max_lost_constr_208: Lost_0[208] = MAX ( Lost_0_expr_208 , 0 )
 max_Q1_constr_208: Q1_var[208] = MAX ( Q_hat_adjusted_208 , 0 )
 min_sales2_constr_208: Sold_1[208] = MIN ( total_stock_second_period_208
   , 708.4006904256271 )
 max_lost2_constr_208: Lost_1[208] = MAX ( Lost_1_expr_208 , 0 )
 GC1254: Fraction[209] = LOGISTIC ( f_var[209] )
 min_sales_constr_209: Sold_0[209] = MIN ( Q0_var[209] , 402.0638126573399 )
 max_lost_constr_209: Lost_0[209] = MAX ( Lost_0_expr_209 , 0 )
 max_Q1_constr_209: Q1_var[209] = MAX ( Q_hat_adjusted_209 , 0 )
 min_sales2_constr_209: Sold_1[209] = MIN ( total_stock_second_period_209
   , 2108.421355066454 )
 max_lost2_constr_209: Lost_1[209] = MAX ( Lost_1_expr_209 , 0 )
 GC1260: Fraction[210] = LOGISTIC ( f_var[210] )
 min_sales_constr_210: Sold_0[210] = MIN ( Q0_var[210] , 130.2593793834359 )
 max_lost_constr_210: Lost_0[210] = MAX ( Lost_0_expr_210 , 0 )
 max_Q1_constr_210: Q1_var[210] = MAX ( Q_hat_adjusted_210 , 0 )
 min_sales2_constr_210: Sold_1[210] = MIN ( total_stock_second_period_210
   , 342.0926388199281 )
 max_lost2_constr_210: Lost_1[210] = MAX ( Lost_1_expr_210 , 0 )
 GC1266: Fraction[211] = LOGISTIC ( f_var[211] )
 min_sales_constr_211: Sold_0[211] = MIN ( Q0_var[211] , 331.8976489154021 )
 max_lost_constr_211: Lost_0[211] = MAX ( Lost_0_expr_211 , 0 )
 max_Q1_constr_211: Q1_var[211] = MAX ( Q_hat_adjusted_211 , 0 )
 min_sales2_constr_211: Sold_1[211] = MIN ( total_stock_second_period_211
   , 2165.498051142852 )
 max_lost2_constr_211: Lost_1[211] = MAX ( Lost_1_expr_211 , 0 )
 GC1272: Fraction[212] = LOGISTIC ( f_var[212] )
 min_sales_constr_212: Sold_0[212] = MIN ( Q0_var[212] , 368.9757114135849 )
 max_lost_constr_212: Lost_0[212] = MAX ( Lost_0_expr_212 , 0 )
 max_Q1_constr_212: Q1_var[212] = MAX ( Q_hat_adjusted_212 , 0 )
 min_sales2_constr_212: Sold_1[212] = MIN ( total_stock_second_period_212
   , 2160.60213973027 )
 max_lost2_constr_212: Lost_1[212] = MAX ( Lost_1_expr_212 , 0 )
 GC1278: Fraction[213] = LOGISTIC ( f_var[213] )
 min_sales_constr_213: Sold_0[213] = MIN ( Q0_var[213] , 63.00705450571935 )
 max_lost_constr_213: Lost_0[213] = MAX ( Lost_0_expr_213 , 0 )
 max_Q1_constr_213: Q1_var[213] = MAX ( Q_hat_adjusted_213 , 0 )
 min_sales2_constr_213: Sold_1[213] = MIN ( total_stock_second_period_213
   , 359.7222933776645 )
 max_lost2_constr_213: Lost_1[213] = MAX ( Lost_1_expr_213 , 0 )
 GC1284: Fraction[214] = LOGISTIC ( f_var[214] )
 min_sales_constr_214: Sold_0[214] = MIN ( Q0_var[214] , 425.7881502388163 )
 max_lost_constr_214: Lost_0[214] = MAX ( Lost_0_expr_214 , 0 )
 max_Q1_constr_214: Q1_var[214] = MAX ( Q_hat_adjusted_214 , 0 )
 min_sales2_constr_214: Sold_1[214] = MIN ( total_stock_second_period_214
   , 1782.476360684366 )
 max_lost2_constr_214: Lost_1[214] = MAX ( Lost_1_expr_214 , 0 )
 GC1290: Fraction[215] = LOGISTIC ( f_var[215] )
 min_sales_constr_215: Sold_0[215] = MIN ( Q0_var[215] , 569.4957627222768 )
 max_lost_constr_215: Lost_0[215] = MAX ( Lost_0_expr_215 , 0 )
 max_Q1_constr_215: Q1_var[215] = MAX ( Q_hat_adjusted_215 , 0 )
 min_sales2_constr_215: Sold_1[215] = MIN ( total_stock_second_period_215
   , 1504.282122464128 )
 max_lost2_constr_215: Lost_1[215] = MAX ( Lost_1_expr_215 , 0 )
 GC1296: Fraction[216] = LOGISTIC ( f_var[216] )
 min_sales_constr_216: Sold_0[216] = MIN ( Q0_var[216] , 290.6983749604484 )
 max_lost_constr_216: Lost_0[216] = MAX ( Lost_0_expr_216 , 0 )
 max_Q1_constr_216: Q1_var[216] = MAX ( Q_hat_adjusted_216 , 0 )
 min_sales2_constr_216: Sold_1[216] = MIN ( total_stock_second_period_216
   , 832.9425905173721 )
 max_lost2_constr_216: Lost_1[216] = MAX ( Lost_1_expr_216 , 0 )
 GC1302: Fraction[217] = LOGISTIC ( f_var[217] )
 min_sales_constr_217: Sold_0[217] = MIN ( Q0_var[217] , 203.3093280319208 )
 max_lost_constr_217: Lost_0[217] = MAX ( Lost_0_expr_217 , 0 )
 max_Q1_constr_217: Q1_var[217] = MAX ( Q_hat_adjusted_217 , 0 )
 min_sales2_constr_217: Sold_1[217] = MIN ( total_stock_second_period_217
   , 555.7965075185953 )
 max_lost2_constr_217: Lost_1[217] = MAX ( Lost_1_expr_217 , 0 )
 GC1308: Fraction[218] = LOGISTIC ( f_var[218] )
 min_sales_constr_218: Sold_0[218] = MIN ( Q0_var[218] , 632.2731906317624 )
 max_lost_constr_218: Lost_0[218] = MAX ( Lost_0_expr_218 , 0 )
 max_Q1_constr_218: Q1_var[218] = MAX ( Q_hat_adjusted_218 , 0 )
 min_sales2_constr_218: Sold_1[218] = MIN ( total_stock_second_period_218
   , 2053.844029307897 )
 max_lost2_constr_218: Lost_1[218] = MAX ( Lost_1_expr_218 , 0 )
 GC1314: Fraction[219] = LOGISTIC ( f_var[219] )
 min_sales_constr_219: Sold_0[219] = MIN ( Q0_var[219] , 354.4425188202355 )
 max_lost_constr_219: Lost_0[219] = MAX ( Lost_0_expr_219 , 0 )
 max_Q1_constr_219: Q1_var[219] = MAX ( Q_hat_adjusted_219 , 0 )
 min_sales2_constr_219: Sold_1[219] = MIN ( total_stock_second_period_219
   , 1356.610419635541 )
 max_lost2_constr_219: Lost_1[219] = MAX ( Lost_1_expr_219 , 0 )
 GC1320: Fraction[220] = LOGISTIC ( f_var[220] )
 min_sales_constr_220: Sold_0[220] = MIN ( Q0_var[220] , 595.6131967865671 )
 max_lost_constr_220: Lost_0[220] = MAX ( Lost_0_expr_220 , 0 )
 max_Q1_constr_220: Q1_var[220] = MAX ( Q_hat_adjusted_220 , 0 )
 min_sales2_constr_220: Sold_1[220] = MIN ( total_stock_second_period_220
   , 2152.986121828997 )
 max_lost2_constr_220: Lost_1[220] = MAX ( Lost_1_expr_220 , 0 )
 GC1326: Fraction[221] = LOGISTIC ( f_var[221] )
 min_sales_constr_221: Sold_0[221] = MIN ( Q0_var[221] , 712.1801777386459 )
 max_lost_constr_221: Lost_0[221] = MAX ( Lost_0_expr_221 , 0 )
 max_Q1_constr_221: Q1_var[221] = MAX ( Q_hat_adjusted_221 , 0 )
 min_sales2_constr_221: Sold_1[221] = MIN ( total_stock_second_period_221
   , 2297.202554404029 )
 max_lost2_constr_221: Lost_1[221] = MAX ( Lost_1_expr_221 , 0 )
 GC1332: Fraction[222] = LOGISTIC ( f_var[222] )
 min_sales_constr_222: Sold_0[222] = MIN ( Q0_var[222] , 406.7527289161615 )
 max_lost_constr_222: Lost_0[222] = MAX ( Lost_0_expr_222 , 0 )
 max_Q1_constr_222: Q1_var[222] = MAX ( Q_hat_adjusted_222 , 0 )
 min_sales2_constr_222: Sold_1[222] = MIN ( total_stock_second_period_222
   , 2516.69482811062 )
 max_lost2_constr_222: Lost_1[222] = MAX ( Lost_1_expr_222 , 0 )
 GC1338: Fraction[223] = LOGISTIC ( f_var[223] )
 min_sales_constr_223: Sold_0[223] = MIN ( Q0_var[223] , 157.1787664812613 )
 max_lost_constr_223: Lost_0[223] = MAX ( Lost_0_expr_223 , 0 )
 max_Q1_constr_223: Q1_var[223] = MAX ( Q_hat_adjusted_223 , 0 )
 min_sales2_constr_223: Sold_1[223] = MIN ( total_stock_second_period_223
   , 1110.656597514951 )
 max_lost2_constr_223: Lost_1[223] = MAX ( Lost_1_expr_223 , 0 )
 GC1344: Fraction[224] = LOGISTIC ( f_var[224] )
 min_sales_constr_224: Sold_0[224] = MIN ( Q0_var[224] , 586.600381579033 )
 max_lost_constr_224: Lost_0[224] = MAX ( Lost_0_expr_224 , 0 )
 max_Q1_constr_224: Q1_var[224] = MAX ( Q_hat_adjusted_224 , 0 )
 min_sales2_constr_224: Sold_1[224] = MIN ( total_stock_second_period_224
   , 1996.744526077786 )
 max_lost2_constr_224: Lost_1[224] = MAX ( Lost_1_expr_224 , 0 )
 GC1350: Fraction[225] = LOGISTIC ( f_var[225] )
 min_sales_constr_225: Sold_0[225] = MIN ( Q0_var[225] , 175.8431246798929 )
 max_lost_constr_225: Lost_0[225] = MAX ( Lost_0_expr_225 , 0 )
 max_Q1_constr_225: Q1_var[225] = MAX ( Q_hat_adjusted_225 , 0 )
 min_sales2_constr_225: Sold_1[225] = MIN ( total_stock_second_period_225
   , 1009.310342243003 )
 max_lost2_constr_225: Lost_1[225] = MAX ( Lost_1_expr_225 , 0 )
 GC1356: Fraction[226] = LOGISTIC ( f_var[226] )
 min_sales_constr_226: Sold_0[226] = MIN ( Q0_var[226] , 324.8468549800301 )
 max_lost_constr_226: Lost_0[226] = MAX ( Lost_0_expr_226 , 0 )
 max_Q1_constr_226: Q1_var[226] = MAX ( Q_hat_adjusted_226 , 0 )
 min_sales2_constr_226: Sold_1[226] = MIN ( total_stock_second_period_226
   , 1703.568464256265 )
 max_lost2_constr_226: Lost_1[226] = MAX ( Lost_1_expr_226 , 0 )
 GC1362: Fraction[227] = LOGISTIC ( f_var[227] )
 min_sales_constr_227: Sold_0[227] = MIN ( Q0_var[227] , 484.0649891446412 )
 max_lost_constr_227: Lost_0[227] = MAX ( Lost_0_expr_227 , 0 )
 max_Q1_constr_227: Q1_var[227] = MAX ( Q_hat_adjusted_227 , 0 )
 min_sales2_constr_227: Sold_1[227] = MIN ( total_stock_second_period_227
   , 1862.01160449829 )
 max_lost2_constr_227: Lost_1[227] = MAX ( Lost_1_expr_227 , 0 )
 GC1368: Fraction[228] = LOGISTIC ( f_var[228] )
 min_sales_constr_228: Sold_0[228] = MIN ( Q0_var[228] , 347.8365761996418 )
 max_lost_constr_228: Lost_0[228] = MAX ( Lost_0_expr_228 , 0 )
 max_Q1_constr_228: Q1_var[228] = MAX ( Q_hat_adjusted_228 , 0 )
 min_sales2_constr_228: Sold_1[228] = MIN ( total_stock_second_period_228
   , 855.3450909852002 )
 max_lost2_constr_228: Lost_1[228] = MAX ( Lost_1_expr_228 , 0 )
 GC1374: Fraction[229] = LOGISTIC ( f_var[229] )
 min_sales_constr_229: Sold_0[229] = MIN ( Q0_var[229] , 471.2941132357026 )
 max_lost_constr_229: Lost_0[229] = MAX ( Lost_0_expr_229 , 0 )
 max_Q1_constr_229: Q1_var[229] = MAX ( Q_hat_adjusted_229 , 0 )
 min_sales2_constr_229: Sold_1[229] = MIN ( total_stock_second_period_229
   , 1268.016638120852 )
 max_lost2_constr_229: Lost_1[229] = MAX ( Lost_1_expr_229 , 0 )
 GC1380: Fraction[230] = LOGISTIC ( f_var[230] )
 min_sales_constr_230: Sold_0[230] = MIN ( Q0_var[230] , 132.9830372801353 )
 max_lost_constr_230: Lost_0[230] = MAX ( Lost_0_expr_230 , 0 )
 max_Q1_constr_230: Q1_var[230] = MAX ( Q_hat_adjusted_230 , 0 )
 min_sales2_constr_230: Sold_1[230] = MIN ( total_stock_second_period_230
   , 614.45685255019 )
 max_lost2_constr_230: Lost_1[230] = MAX ( Lost_1_expr_230 , 0 )
 GC1386: Fraction[231] = LOGISTIC ( f_var[231] )
 min_sales_constr_231: Sold_0[231] = MIN ( Q0_var[231] , 101.1916516040443 )
 max_lost_constr_231: Lost_0[231] = MAX ( Lost_0_expr_231 , 0 )
 max_Q1_constr_231: Q1_var[231] = MAX ( Q_hat_adjusted_231 , 0 )
 min_sales2_constr_231: Sold_1[231] = MIN ( total_stock_second_period_231
   , 803.9628061026388 )
 max_lost2_constr_231: Lost_1[231] = MAX ( Lost_1_expr_231 , 0 )
 GC1392: Fraction[232] = LOGISTIC ( f_var[232] )
 min_sales_constr_232: Sold_0[232] = MIN ( Q0_var[232] , 268.7082871069592 )
 max_lost_constr_232: Lost_0[232] = MAX ( Lost_0_expr_232 , 0 )
 max_Q1_constr_232: Q1_var[232] = MAX ( Q_hat_adjusted_232 , 0 )
 min_sales2_constr_232: Sold_1[232] = MIN ( total_stock_second_period_232
   , 897.8934453526914 )
 max_lost2_constr_232: Lost_1[232] = MAX ( Lost_1_expr_232 , 0 )
 GC1398: Fraction[233] = LOGISTIC ( f_var[233] )
 min_sales_constr_233: Sold_0[233] = MIN ( Q0_var[233] , 277.0384014393732 )
 max_lost_constr_233: Lost_0[233] = MAX ( Lost_0_expr_233 , 0 )
 max_Q1_constr_233: Q1_var[233] = MAX ( Q_hat_adjusted_233 , 0 )
 min_sales2_constr_233: Sold_1[233] = MIN ( total_stock_second_period_233
   , 944.7777870953657 )
 max_lost2_constr_233: Lost_1[233] = MAX ( Lost_1_expr_233 , 0 )
 GC1404: Fraction[234] = LOGISTIC ( f_var[234] )
 min_sales_constr_234: Sold_0[234] = MIN ( Q0_var[234] , 160.0402907632731 )
 max_lost_constr_234: Lost_0[234] = MAX ( Lost_0_expr_234 , 0 )
 max_Q1_constr_234: Q1_var[234] = MAX ( Q_hat_adjusted_234 , 0 )
 min_sales2_constr_234: Sold_1[234] = MIN ( total_stock_second_period_234
   , 679.1398069850899 )
 max_lost2_constr_234: Lost_1[234] = MAX ( Lost_1_expr_234 , 0 )
 GC1410: Fraction[235] = LOGISTIC ( f_var[235] )
 min_sales_constr_235: Sold_0[235] = MIN ( Q0_var[235] , 156.2134333933114 )
 max_lost_constr_235: Lost_0[235] = MAX ( Lost_0_expr_235 , 0 )
 max_Q1_constr_235: Q1_var[235] = MAX ( Q_hat_adjusted_235 , 0 )
 min_sales2_constr_235: Sold_1[235] = MIN ( total_stock_second_period_235
   , 1285.29531811116 )
 max_lost2_constr_235: Lost_1[235] = MAX ( Lost_1_expr_235 , 0 )
 GC1416: Fraction[236] = LOGISTIC ( f_var[236] )
 min_sales_constr_236: Sold_0[236] = MIN ( Q0_var[236] , 558.9120703340504 )
 max_lost_constr_236: Lost_0[236] = MAX ( Lost_0_expr_236 , 0 )
 max_Q1_constr_236: Q1_var[236] = MAX ( Q_hat_adjusted_236 , 0 )
 min_sales2_constr_236: Sold_1[236] = MIN ( total_stock_second_period_236
   , 1624.548751157519 )
 max_lost2_constr_236: Lost_1[236] = MAX ( Lost_1_expr_236 , 0 )
 GC1422: Fraction[237] = LOGISTIC ( f_var[237] )
 min_sales_constr_237: Sold_0[237] = MIN ( Q0_var[237] , 298.9692339494929 )
 max_lost_constr_237: Lost_0[237] = MAX ( Lost_0_expr_237 , 0 )
 max_Q1_constr_237: Q1_var[237] = MAX ( Q_hat_adjusted_237 , 0 )
 min_sales2_constr_237: Sold_1[237] = MIN ( total_stock_second_period_237
   , 1473.714354490466 )
 max_lost2_constr_237: Lost_1[237] = MAX ( Lost_1_expr_237 , 0 )
 GC1428: Fraction[238] = LOGISTIC ( f_var[238] )
 min_sales_constr_238: Sold_0[238] = MIN ( Q0_var[238] , 272.8704498859149 )
 max_lost_constr_238: Lost_0[238] = MAX ( Lost_0_expr_238 , 0 )
 max_Q1_constr_238: Q1_var[238] = MAX ( Q_hat_adjusted_238 , 0 )
 min_sales2_constr_238: Sold_1[238] = MIN ( total_stock_second_period_238
   , 593.3703776561499 )
 max_lost2_constr_238: Lost_1[238] = MAX ( Lost_1_expr_238 , 0 )
 GC1434: Fraction[239] = LOGISTIC ( f_var[239] )
 min_sales_constr_239: Sold_0[239] = MIN ( Q0_var[239] , 65.62455546577618 )
 max_lost_constr_239: Lost_0[239] = MAX ( Lost_0_expr_239 , 0 )
 max_Q1_constr_239: Q1_var[239] = MAX ( Q_hat_adjusted_239 , 0 )
 min_sales2_constr_239: Sold_1[239] = MIN ( total_stock_second_period_239
   , 435.0255291043418 )
 max_lost2_constr_239: Lost_1[239] = MAX ( Lost_1_expr_239 , 0 )
 GC1440: Fraction[240] = LOGISTIC ( f_var[240] )
 min_sales_constr_240: Sold_0[240] = MIN ( Q0_var[240] , 242.5892901516935 )
 max_lost_constr_240: Lost_0[240] = MAX ( Lost_0_expr_240 , 0 )
 max_Q1_constr_240: Q1_var[240] = MAX ( Q_hat_adjusted_240 , 0 )
 min_sales2_constr_240: Sold_1[240] = MIN ( total_stock_second_period_240
   , 1000.828354995933 )
 max_lost2_constr_240: Lost_1[240] = MAX ( Lost_1_expr_240 , 0 )
 GC1446: Fraction[241] = LOGISTIC ( f_var[241] )
 min_sales_constr_241: Sold_0[241] = MIN ( Q0_var[241] , 180.6812511953311 )
 max_lost_constr_241: Lost_0[241] = MAX ( Lost_0_expr_241 , 0 )
 max_Q1_constr_241: Q1_var[241] = MAX ( Q_hat_adjusted_241 , 0 )
 min_sales2_constr_241: Sold_1[241] = MIN ( total_stock_second_period_241
   , 612.2469752146003 )
 max_lost2_constr_241: Lost_1[241] = MAX ( Lost_1_expr_241 , 0 )
 GC1452: Fraction[242] = LOGISTIC ( f_var[242] )
 min_sales_constr_242: Sold_0[242] = MIN ( Q0_var[242] , 587.8996746666605 )
 max_lost_constr_242: Lost_0[242] = MAX ( Lost_0_expr_242 , 0 )
 max_Q1_constr_242: Q1_var[242] = MAX ( Q_hat_adjusted_242 , 0 )
 min_sales2_constr_242: Sold_1[242] = MIN ( total_stock_second_period_242
   , 2523.588319787366 )
 max_lost2_constr_242: Lost_1[242] = MAX ( Lost_1_expr_242 , 0 )
 GC1458: Fraction[243] = LOGISTIC ( f_var[243] )
 min_sales_constr_243: Sold_0[243] = MIN ( Q0_var[243] , 217.72846961345 )
 max_lost_constr_243: Lost_0[243] = MAX ( Lost_0_expr_243 , 0 )
 max_Q1_constr_243: Q1_var[243] = MAX ( Q_hat_adjusted_243 , 0 )
 min_sales2_constr_243: Sold_1[243] = MIN ( total_stock_second_period_243
   , 974.3888331272441 )
 max_lost2_constr_243: Lost_1[243] = MAX ( Lost_1_expr_243 , 0 )
 GC1464: Fraction[244] = LOGISTIC ( f_var[244] )
 min_sales_constr_244: Sold_0[244] = MIN ( Q0_var[244] , 403.1149483496066 )
 max_lost_constr_244: Lost_0[244] = MAX ( Lost_0_expr_244 , 0 )
 max_Q1_constr_244: Q1_var[244] = MAX ( Q_hat_adjusted_244 , 0 )
 min_sales2_constr_244: Sold_1[244] = MIN ( total_stock_second_period_244
   , 2039.707295387194 )
 max_lost2_constr_244: Lost_1[244] = MAX ( Lost_1_expr_244 , 0 )
 GC1470: Fraction[245] = LOGISTIC ( f_var[245] )
 min_sales_constr_245: Sold_0[245] = MIN ( Q0_var[245] , 209.0748380809555 )
 max_lost_constr_245: Lost_0[245] = MAX ( Lost_0_expr_245 , 0 )
 max_Q1_constr_245: Q1_var[245] = MAX ( Q_hat_adjusted_245 , 0 )
 min_sales2_constr_245: Sold_1[245] = MIN ( total_stock_second_period_245
   , 1516.009586511807 )
 max_lost2_constr_245: Lost_1[245] = MAX ( Lost_1_expr_245 , 0 )
 GC1476: Fraction[246] = LOGISTIC ( f_var[246] )
 min_sales_constr_246: Sold_0[246] = MIN ( Q0_var[246] , 448.1706425341932 )
 max_lost_constr_246: Lost_0[246] = MAX ( Lost_0_expr_246 , 0 )
 max_Q1_constr_246: Q1_var[246] = MAX ( Q_hat_adjusted_246 , 0 )
 min_sales2_constr_246: Sold_1[246] = MIN ( total_stock_second_period_246
   , 1292.533039627259 )
 max_lost2_constr_246: Lost_1[246] = MAX ( Lost_1_expr_246 , 0 )
 GC1482: Fraction[247] = LOGISTIC ( f_var[247] )
 min_sales_constr_247: Sold_0[247] = MIN ( Q0_var[247] , 523.2013456654935 )
 max_lost_constr_247: Lost_0[247] = MAX ( Lost_0_expr_247 , 0 )
 max_Q1_constr_247: Q1_var[247] = MAX ( Q_hat_adjusted_247 , 0 )
 min_sales2_constr_247: Sold_1[247] = MIN ( total_stock_second_period_247
   , 1671.612124024781 )
 max_lost2_constr_247: Lost_1[247] = MAX ( Lost_1_expr_247 , 0 )
 GC1488: Fraction[248] = LOGISTIC ( f_var[248] )
 min_sales_constr_248: Sold_0[248] = MIN ( Q0_var[248] , 533.2026437640332 )
 max_lost_constr_248: Lost_0[248] = MAX ( Lost_0_expr_248 , 0 )
 max_Q1_constr_248: Q1_var[248] = MAX ( Q_hat_adjusted_248 , 0 )
 min_sales2_constr_248: Sold_1[248] = MIN ( total_stock_second_period_248
   , 2283.740604754639 )
 max_lost2_constr_248: Lost_1[248] = MAX ( Lost_1_expr_248 , 0 )
 GC1494: Fraction[249] = LOGISTIC ( f_var[249] )
 min_sales_constr_249: Sold_0[249] = MIN ( Q0_var[249] , 199.9707338281307 )
 max_lost_constr_249: Lost_0[249] = MAX ( Lost_0_expr_249 , 0 )
 max_Q1_constr_249: Q1_var[249] = MAX ( Q_hat_adjusted_249 , 0 )
 min_sales2_constr_249: Sold_1[249] = MIN ( total_stock_second_period_249
   , 916.6298460252046 )
 max_lost2_constr_249: Lost_1[249] = MAX ( Lost_1_expr_249 , 0 )
End
