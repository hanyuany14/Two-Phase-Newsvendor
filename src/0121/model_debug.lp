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
   + profits_vars[98] + profits_vars[99]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: 0.331321951038668 alphas[0] + 0.428261078213753 alphas[1] - alphas[2]
   + f_var[0] = 0
 R3: - 196.8542016893929 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 171.9705526854737
 R5: Q0_var[0] + Q_hat_adjusted_0 = 207.1379285830255
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 40.00056744336146
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 500 Sold_0[0] - 500 Sold_1[0] + 500 Lost_0[0] + 500 Lost_1[0]
   + 500 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: 1.020232202694403 alphas[0] + 0.31186518271054 alphas[1] - alphas[2]
   + f_var[1] = 0
 R13: - 196.8542016893929 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 124.9174642130688
 R15: Q0_var[1] + Q_hat_adjusted_1 = 178.6606743639977
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 56.96688789215834
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 500 Sold_0[1] - 500 Sold_1[1] + 500 Lost_0[1] + 500 Lost_1[1]
   + 500 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.2377563398324784 alphas[0] + 0.5658390468687988 alphas[1]
   - alphas[2] + f_var[2] = 0
 R23: - 196.8542016893929 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 178.7053681962325
 R25: Q0_var[2] + Q_hat_adjusted_2 = 217.3750840595637
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 35.96797259406476
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 500 Sold_0[2] - 500 Sold_1[2] + 500 Lost_0[2] + 500 Lost_1[2]
   + 500 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.1631075852883452 alphas[0] + 0.7284876180131219 alphas[1]
   - alphas[2] + f_var[3] = 0
 R33: - 196.8542016893929 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 137.3831500547491
 R35: Q0_var[3] + Q_hat_adjusted_3 = 158.7545784426345
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 21.56865094348434
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 500 Sold_0[3] - 500 Sold_1[3] + 500 Lost_0[3] + 500 Lost_1[3]
   + 500 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.7424809111061429 alphas[0] - 0.0121439627165657 alphas[1]
   - alphas[2] + f_var[4] = 0
 R43: - 196.8542016893929 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 143.9535066724153
 R45: Q0_var[4] + Q_hat_adjusted_4 = 158.8444443253275
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 6.943726725029627
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 500 Sold_0[4] - 500 Sold_1[4] + 500 Lost_0[4] + 500 Lost_1[4]
   + 500 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: 0.1918845722311603 alphas[0] - 0.9745260177839964 alphas[1]
   - alphas[2] + f_var[5] = 0
 R53: - 196.8542016893929 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 199.2437125823047
 R55: Q0_var[5] + Q_hat_adjusted_5 = 229.6491021728086
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 35.59143750066534
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 500 Sold_0[5] - 500 Sold_1[5] + 500 Lost_0[5] + 500 Lost_1[5]
   + 500 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 1.11327540045227 alphas[0] + 1.053209923414298 alphas[1] - alphas[2]
   + f_var[6] = 0
 R63: - 196.8542016893929 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 159.6723925421878
 R65: Q0_var[6] + Q_hat_adjusted_6 = 180.4457452583764
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 25.57223421387051
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 500 Sold_0[6] - 500 Sold_1[6] + 500 Lost_0[6] + 500 Lost_1[6]
   + 500 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: 1.066527563861114 alphas[0] - 0.6008595762617773 alphas[1]
   - alphas[2] + f_var[7] = 0
 R73: - 196.8542016893929 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 171.2283386759027
 R75: Q0_var[7] + Q_hat_adjusted_7 = 194.4889104911137
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 17.69412214989524
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 500 Sold_0[7] - 500 Sold_1[7] + 500 Lost_0[7] + 500 Lost_1[7]
   + 500 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: 0.4961149135233016 alphas[0] + 0.4046911371094515 alphas[1]
   - alphas[2] + f_var[8] = 0
 R83: - 196.8542016893929 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 132.0321983449813
 R85: Q0_var[8] + Q_hat_adjusted_8 = 163.4676884495768
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 22.07722047943105
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 500 Sold_0[8] - 500 Sold_1[8] + 500 Lost_0[8] + 500 Lost_1[8]
   + 500 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.2271206102846862 alphas[0] + 1.449597465848972 alphas[1]
   - alphas[2] + f_var[9] = 0
 R93: - 196.8542016893929 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 150.9931699477943
 R95: Q0_var[9] + Q_hat_adjusted_9 = 169.9739670811738
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 20.74203343463805
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 500 Sold_0[9] - 500 Sold_1[9] + 500 Lost_0[9] + 500 Lost_1[9]
   + 500 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: 1.417315971474846 alphas[0] + 0.083222700215112 alphas[1]
   - alphas[2] + f_var[10] = 0
 R103: - 196.8542016893929 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 109.7827951535821
 R105: Q0_var[10] + Q_hat_adjusted_10 = 139.8790360778002
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 25.86372640942696
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 500 Sold_0[10] - 500 Sold_1[10] + 500 Lost_0[10] + 500 Lost_1[10]
   + 500 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 0.4830840751918974 alphas[0] + 0.0742074002619241 alphas[1]
   - alphas[2] + f_var[11] = 0
 R113: - 196.8542016893929 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 156.9217355903971
 R115: Q0_var[11] + Q_hat_adjusted_11 = 182.1306958613634
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 31.05309535586031
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 500 Sold_0[11] - 500 Sold_1[11] + 500 Lost_0[11] + 500 Lost_1[11]
   + 500 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: 0.6516374869143376 alphas[0] + 1.348200581756968 alphas[1]
   - alphas[2] + f_var[12] = 0
 R123: - 196.8542016893929 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 110.2675355014723
 R125: Q0_var[12] + Q_hat_adjusted_12 = 133.8320901030839
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 22.23244107479613
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 500 Sold_0[12] - 500 Sold_1[12] + 500 Lost_0[12] + 500 Lost_1[12]
   + 500 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: 0.8263226117674036 alphas[0] + 1.335976558371678 alphas[1]
   - alphas[2] + f_var[13] = 0
 R133: - 196.8542016893929 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 133.5567173791324
 R135: Q0_var[13] + Q_hat_adjusted_13 = 174.1812449700533
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 43.48610893436775
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 500 Sold_0[13] - 500 Sold_1[13] + 500 Lost_0[13] + 500 Lost_1[13]
   + 500 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: - 1.478453476246146 alphas[0] + 0.9084217144790343 alphas[1]
   - alphas[2] + f_var[14] = 0
 R143: - 196.8542016893929 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 181.912974906696
 R145: Q0_var[14] + Q_hat_adjusted_14 = 219.7206570997228
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 37.40275210109417
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 500 Sold_0[14] - 500 Sold_1[14] + 500 Lost_0[14] + 500 Lost_1[14]
   + 500 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 0.1694818324133431 alphas[0] - 1.469719944581323 alphas[1]
   - alphas[2] + f_var[15] = 0
 R153: - 196.8542016893929 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 140.5069759576123
 R155: Q0_var[15] + Q_hat_adjusted_15 = 166.0415072969981
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 22.1281266149289
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 500 Sold_0[15] - 500 Sold_1[15] + 500 Lost_0[15] + 500 Lost_1[15]
   + 500 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: - 0.9720245184241577 alphas[0] - 1.076480378944146 alphas[1]
   - alphas[2] + f_var[16] = 0
 R163: - 196.8542016893929 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 193.5398129351872
 R165: Q0_var[16] + Q_hat_adjusted_16 = 236.5647210062642
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 38.77726993647558
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 500 Sold_0[16] - 500 Sold_1[16] + 500 Lost_0[16] + 500 Lost_1[16]
   + 500 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 0.1170938994553131 alphas[0] - 1.141355107155857 alphas[1]
   - alphas[2] + f_var[17] = 0
 R173: - 196.8542016893929 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 163.2283874502225
 R175: Q0_var[17] + Q_hat_adjusted_17 = 207.3835750612853
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 47.98179273765295
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 500 Sold_0[17] - 500 Sold_1[17] + 500 Lost_0[17] + 500 Lost_1[17]
   + 500 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.3099360026136093 alphas[0] - 0.9998672115895263 alphas[1]
   - alphas[2] + f_var[18] = 0
 R183: - 196.8542016893929 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 181.0505986101853
 R185: Q0_var[18] + Q_hat_adjusted_18 = 183.16406529943
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 4.477984455359557
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 500 Sold_0[18] - 500 Sold_1[18] + 500 Lost_0[18] + 500 Lost_1[18]
   + 500 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.663551677434663 alphas[0] - 0.4489095829753688 alphas[1]
   - alphas[2] + f_var[19] = 0
 R193: - 196.8542016893929 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 204.9390731849088
 R195: Q0_var[19] + Q_hat_adjusted_19 = 241.9526444670185
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 39.81206344818219
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 500 Sold_0[19] - 500 Sold_1[19] + 500 Lost_0[19] + 500 Lost_1[19]
   + 500 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: 1.448627032216275 alphas[0] - 0.9292317303749468 alphas[1]
   - alphas[2] + f_var[20] = 0
 R203: - 196.8542016893929 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 137.1546967403893
 R205: Q0_var[20] + Q_hat_adjusted_20 = 186.2713792095684
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 53.64953356637277
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 500 Sold_0[20] - 500 Sold_1[20] + 500 Lost_0[20] + 500 Lost_1[20]
   + 500 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: 0.7189722240815695 alphas[0] - 0.454351734672241 alphas[1]
   - alphas[2] + f_var[21] = 0
 R213: - 196.8542016893929 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 153.1190901099322
 R215: Q0_var[21] + Q_hat_adjusted_21 = 184.437072353724
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 38.21243883440913
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 500 Sold_0[21] - 500 Sold_1[21] + 500 Lost_0[21] + 500 Lost_1[21]
   + 500 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.8843999226205789 alphas[0] + 0.3741938275676893 alphas[1]
   - alphas[2] + f_var[22] = 0
 R223: - 196.8542016893929 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 132.4254014425122
 R225: Q0_var[22] + Q_hat_adjusted_22 = 160.3164420821448
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 28.18128674784869
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 500 Sold_0[22] - 500 Sold_1[22] + 500 Lost_0[22] + 500 Lost_1[22]
   + 500 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 1.697608008993154 alphas[0] - 0.9550175732314641 alphas[1]
   - alphas[2] + f_var[23] = 0
 R233: - 196.8542016893929 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 171.474319846935
 R235: Q0_var[23] + Q_hat_adjusted_23 = 215.0979460236314
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 56.63239264835586
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 500 Sold_0[23] - 500 Sold_1[23] + 500 Lost_0[23] + 500 Lost_1[23]
   + 500 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: - 0.6134557928614575 alphas[0] - 1.479265540601661 alphas[1]
   - alphas[2] + f_var[24] = 0
 R243: - 196.8542016893929 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 170.1947067262712
 R245: Q0_var[24] + Q_hat_adjusted_24 = 205.3765463715812
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 34.19186156250329
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 500 Sold_0[24] - 500 Sold_1[24] + 500 Lost_0[24] + 500 Lost_1[24]
   + 500 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: 0.765518404628077 alphas[0] - 1.608525887006107 alphas[1]
   - alphas[2] + f_var[25] = 0
 R253: - 196.8542016893929 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 171.2695199646037
 R255: Q0_var[25] + Q_hat_adjusted_25 = 211.576100760753
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 33.48575890842393
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 500 Sold_0[25] - 500 Sold_1[25] + 500 Lost_0[25] + 500 Lost_1[25]
   + 500 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: - 0.149386876969774 alphas[0] + 1.21918629430375 alphas[1]
   - alphas[2] + f_var[26] = 0
 R263: - 196.8542016893929 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 148.0716584518483
 R265: Q0_var[26] + Q_hat_adjusted_26 = 192.5888564854131
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 47.71958188970855
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 500 Sold_0[26] - 500 Sold_1[26] + 500 Lost_0[26] + 500 Lost_1[26]
   + 500 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.323127349106093 alphas[0] + 0.3248304334703782 alphas[1]
   - alphas[2] + f_var[27] = 0
 R273: - 196.8542016893929 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 197.746021833302
 R275: Q0_var[27] + Q_hat_adjusted_27 = 224.9348003018988
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 27.85371514600896
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 500 Sold_0[27] - 500 Sold_1[27] + 500 Lost_0[27] + 500 Lost_1[27]
   + 500 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: - 1.652740651797471 alphas[0] - 1.388269017244192 alphas[1]
   - alphas[2] + f_var[28] = 0
 R283: - 196.8542016893929 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 214.6649974676589
 R285: Q0_var[28] + Q_hat_adjusted_28 = 256.3776853849249
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 43.80366275174308
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 500 Sold_0[28] - 500 Sold_1[28] + 500 Lost_0[28] + 500 Lost_1[28]
   + 500 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 1.550215307448353 alphas[0] + 1.825265242192512 alphas[1]
   - alphas[2] + f_var[29] = 0
 R293: - 196.8542016893929 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 106.0300190055475
 R295: Q0_var[29] + Q_hat_adjusted_29 = 130.4827711769744
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 27.66386612557105
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 500 Sold_0[29] - 500 Sold_1[29] + 500 Lost_0[29] + 500 Lost_1[29]
   + 500 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 0.2251822124206934 alphas[0] - 0.6360354175279859 alphas[1]
   - alphas[2] + f_var[30] = 0
 R303: - 196.8542016893929 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 160.9523400246461
 R305: Q0_var[30] + Q_hat_adjusted_30 = 206.9534766281211
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 46.90148567368223
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 500 Sold_0[30] - 500 Sold_1[30] + 500 Lost_0[30] + 500 Lost_1[30]
   + 500 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: - 1.172462221809449 alphas[0] + 1.648474259133731 alphas[1]
   - alphas[2] + f_var[31] = 0
 R313: - 196.8542016893929 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 170.2989269211271
 R315: Q0_var[31] + Q_hat_adjusted_31 = 186.4314235802316
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 14.90864950417645
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 500 Sold_0[31] - 500 Sold_1[31] + 500 Lost_0[31] + 500 Lost_1[31]
   + 500 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: - 0.2252051294042313 alphas[0] + 0.5969427881186808 alphas[1]
   - alphas[2] + f_var[32] = 0
 R323: - 196.8542016893929 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 158.755917130324
 R325: Q0_var[32] + Q_hat_adjusted_32 = 186.3946224249084
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 23.64679122005043
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 500 Sold_0[32] - 500 Sold_1[32] + 500 Lost_0[32] + 500 Lost_1[32]
   + 500 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: 0.00267643154120212 alphas[0] + 1.397811436151692 alphas[1]
   - alphas[2] + f_var[33] = 0
 R333: - 196.8542016893929 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 145.6860643563938
 R335: Q0_var[33] + Q_hat_adjusted_33 = 176.4059336049086
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 25.53716001953845
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 500 Sold_0[33] - 500 Sold_1[33] + 500 Lost_0[33] + 500 Lost_1[33]
   + 500 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: - 1.207593170630001 alphas[0] - 0.6434584600189068 alphas[1]
   - alphas[2] + f_var[34] = 0
 R343: - 196.8542016893929 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 198.5514384897298
 R345: Q0_var[34] + Q_hat_adjusted_34 = 235.7161473972949
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 41.5614366355288
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 500 Sold_0[34] - 500 Sold_1[34] + 500 Lost_0[34] + 500 Lost_1[34]
   + 500 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: 1.563101149355713 alphas[0] + 0.6693756552195055 alphas[1]
   - alphas[2] + f_var[35] = 0
 R353: - 196.8542016893929 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 109.7540047812708
 R355: Q0_var[35] + Q_hat_adjusted_35 = 142.9865872565977
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 32.22749803008396
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 500 Sold_0[35] - 500 Sold_1[35] + 500 Lost_0[35] + 500 Lost_1[35]
   + 500 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 1.065547362299753 alphas[0] + 0.8554375311244723 alphas[1]
   - alphas[2] + f_var[36] = 0
 R363: - 196.8542016893929 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 135.5815287331823
 R365: Q0_var[36] + Q_hat_adjusted_36 = 176.5510231163892
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 39.65949512585815
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 500 Sold_0[36] - 500 Sold_1[36] + 500 Lost_0[36] + 500 Lost_1[36]
   + 500 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.5771280106919855 alphas[0] + 1.377701728894893 alphas[1]
   - alphas[2] + f_var[37] = 0
 R373: - 196.8542016893929 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 187.5707800870579
 R375: Q0_var[37] + Q_hat_adjusted_37 = 246.3524501565681
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 58.15954461047021
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 500 Sold_0[37] - 500 Sold_1[37] + 500 Lost_0[37] + 500 Lost_1[37]
   + 500 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: 0.2362506931940236 alphas[0] + 1.19829270246144 alphas[1]
   - alphas[2] + f_var[38] = 0
 R383: - 196.8542016893929 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 164.5066144839163
 R385: Q0_var[38] + Q_hat_adjusted_38 = 207.1611934477997
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 39.14235574889312
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 500 Sold_0[38] - 500 Sold_1[38] + 500 Lost_0[38] + 500 Lost_1[38]
   + 500 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 0.0935720210395559 alphas[0] + 1.080753196992097 alphas[1]
   - alphas[2] + f_var[39] = 0
 R393: - 196.8542016893929 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 120.5427379818348
 R395: Q0_var[39] + Q_hat_adjusted_39 = 138.8249225929803
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 20.55161373790484
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 500 Sold_0[39] - 500 Sold_1[39] + 500 Lost_0[39] + 500 Lost_1[39]
   + 500 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: - 0.6648258864345039 alphas[0] - 0.8129181828923424 alphas[1]
   - alphas[2] + f_var[40] = 0
 R403: - 196.8542016893929 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 179.6732770887298
 R405: Q0_var[40] + Q_hat_adjusted_40 = 206.7214710555481
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 36.88608347891664
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 500 Sold_0[40] - 500 Sold_1[40] + 500 Lost_0[40] + 500 Lost_1[40]
   + 500 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: - 0.8137599802679647 alphas[0] + 0.6492389316738829 alphas[1]
   - alphas[2] + f_var[41] = 0
 R413: - 196.8542016893929 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 153.8877293941184
 R415: Q0_var[41] + Q_hat_adjusted_41 = 199.642585350641
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 45.95059525813855
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 500 Sold_0[41] - 500 Sold_1[41] + 500 Lost_0[41] + 500 Lost_1[41]
   + 500 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.4033644270600876 alphas[0] + 0.3094404357239788 alphas[1]
   - alphas[2] + f_var[42] = 0
 R423: - 196.8542016893929 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 197.4094493607441
 R425: Q0_var[42] + Q_hat_adjusted_42 = 244.726539715314
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 47.99268709071016
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 500 Sold_0[42] - 500 Sold_1[42] + 500 Lost_0[42] + 500 Lost_1[42]
   + 500 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: 0.8685562107382476 alphas[0] - 0.283703738997661 alphas[1]
   - alphas[2] + f_var[43] = 0
 R433: - 196.8542016893929 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 130.0818736708006
 R435: Q0_var[43] + Q_hat_adjusted_43 = 175.7927033628874
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 47.66134363513352
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 500 Sold_0[43] - 500 Sold_1[43] + 500 Lost_0[43] + 500 Lost_1[43]
   + 500 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: 0.4655788594230167 alphas[0] - 1.028618022533617 alphas[1]
   - alphas[2] + f_var[44] = 0
 R443: - 196.8542016893929 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 159.2351471373275
 R445: Q0_var[44] + Q_hat_adjusted_44 = 210.1524826317359
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 58.84888756857456
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 500 Sold_0[44] - 500 Sold_1[44] + 500 Lost_0[44] + 500 Lost_1[44]
   + 500 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: - 0.7439239410690743 alphas[0] - 0.6989566283797322 alphas[1]
   - alphas[2] + f_var[45] = 0
 R453: - 196.8542016893929 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 187.8292467518336
 R455: Q0_var[45] + Q_hat_adjusted_45 = 217.7160691500758
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 27.92334495927061
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 500 Sold_0[45] - 500 Sold_1[45] + 500 Lost_0[45] + 500 Lost_1[45]
   + 500 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: - 0.7790604462312499 alphas[0] + 0.882417128375488 alphas[1]
   - alphas[2] + f_var[46] = 0
 R463: - 196.8542016893929 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 188.1533080893215
 R465: Q0_var[46] + Q_hat_adjusted_46 = 231.0014124226525
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 35.33877421277447
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 500 Sold_0[46] - 500 Sold_1[46] + 500 Lost_0[46] + 500 Lost_1[46]
   + 500 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: - 0.7678938873537752 alphas[0] - 0.1327358843982813 alphas[1]
   - alphas[2] + f_var[47] = 0
 R473: - 196.8542016893929 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 159.8675131459254
 R475: Q0_var[47] + Q_hat_adjusted_47 = 191.5813567549554
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 37.24957815626567
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 500 Sold_0[47] - 500 Sold_1[47] + 500 Lost_0[47] + 500 Lost_1[47]
   + 500 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: - 0.6826285558956562 alphas[0] - 0.034639200887802 alphas[1]
   - alphas[2] + f_var[48] = 0
 R483: - 196.8542016893929 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 128.7438596425112
 R485: Q0_var[48] + Q_hat_adjusted_48 = 163.228902246521
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 36.6554026870744
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 500 Sold_0[48] - 500 Sold_1[48] + 500 Lost_0[48] + 500 Lost_1[48]
   + 500 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: - 0.2585646049856958 alphas[0] + 1.023821526556555 alphas[1]
   - alphas[2] + f_var[49] = 0
 R493: - 196.8542016893929 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 145.7571018959608
 R495: Q0_var[49] + Q_hat_adjusted_49 = 180.8325935328482
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 41.38627351476885
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 500 Sold_0[49] - 500 Sold_1[49] + 500 Lost_0[49] + 500 Lost_1[49]
   + 500 Left_1[49] + profits_vars[49] = 0
 Fraction_lower_bound_50: Fraction[50] >= 0
 Fraction_upper_bound_50: Fraction[50] <= 1
 R502: 1.279855976502718 alphas[0] + 1.674856212493542 alphas[1]
   - alphas[2] + f_var[50] = 0
 R503: - 196.8542016893929 Fraction[50] + Q0_var[50] = 0
 R504: Q0_var[50] + Lost_0_expr_50 = 112.1953665086659
 R505: Q0_var[50] + Q_hat_adjusted_50 = 150.4812042528149
 R506: Sold_0[50] - Q0_var[50] - Q1_var[50] + total_stock_second_period_50
   = 0
 R507: total_stock_second_period_50 + Lost_1_expr_50 = 33.58445448056638
 R508: Sold_1[50] + Left_1[50] - total_stock_second_period_50 = 0
 R509: - 500 Sold_0[50] - 500 Sold_1[50] + 500 Lost_0[50] + 500 Lost_1[50]
   + 500 Left_1[50] + profits_vars[50] = 0
 Fraction_lower_bound_51: Fraction[51] >= 0
 Fraction_upper_bound_51: Fraction[51] <= 1
 R512: 1.688135098997252 alphas[0] - 0.4751876823669151 alphas[1]
   - alphas[2] + f_var[51] = 0
 R513: - 196.8542016893929 Fraction[51] + Q0_var[51] = 0
 R514: Q0_var[51] + Lost_0_expr_51 = 158.4857945539278
 R515: Q0_var[51] + Q_hat_adjusted_51 = 168.2998722240272
 R516: Sold_0[51] - Q0_var[51] - Q1_var[51] + total_stock_second_period_51
   = 0
 R517: total_stock_second_period_51 + Lost_1_expr_51 = 8.350276965676997
 R518: Sold_1[51] + Left_1[51] - total_stock_second_period_51 = 0
 R519: - 500 Sold_0[51] - 500 Sold_1[51] + 500 Lost_0[51] + 500 Lost_1[51]
   + 500 Left_1[51] + profits_vars[51] = 0
 Fraction_lower_bound_52: Fraction[52] >= 0
 Fraction_upper_bound_52: Fraction[52] <= 1
 R522: - 1.153872487510526 alphas[0] + 0.4624552989480549 alphas[1]
   - alphas[2] + f_var[52] = 0
 R523: - 196.8542016893929 Fraction[52] + Q0_var[52] = 0
 R524: Q0_var[52] + Lost_0_expr_52 = 195.4216836397702
 R525: Q0_var[52] + Q_hat_adjusted_52 = 236.4276490329655
 R526: Sold_0[52] - Q0_var[52] - Q1_var[52] + total_stock_second_period_52
   = 0
 R527: total_stock_second_period_52 + Lost_1_expr_52 = 42.40169722335882
 R528: Sold_1[52] + Left_1[52] - total_stock_second_period_52 = 0
 R529: - 500 Sold_0[52] - 500 Sold_1[52] + 500 Lost_0[52] + 500 Lost_1[52]
   + 500 Left_1[52] + profits_vars[52] = 0
 Fraction_lower_bound_53: Fraction[53] >= 0
 Fraction_upper_bound_53: Fraction[53] <= 1
 R532: - 1.541819959863152 alphas[0] + 0.8553164878394202 alphas[1]
   - alphas[2] + f_var[53] = 0
 R533: - 196.8542016893929 Fraction[53] + Q0_var[53] = 0
 R534: Q0_var[53] + Lost_0_expr_53 = 162.6858902368283
 R535: Q0_var[53] + Q_hat_adjusted_53 = 210.1810731086608
 R536: Sold_0[53] - Q0_var[53] - Q1_var[53] + total_stock_second_period_53
   = 0
 R537: total_stock_second_period_53 + Lost_1_expr_53 = 42.47178348039976
 R538: Sold_1[53] + Left_1[53] - total_stock_second_period_53 = 0
 R539: - 500 Sold_0[53] - 500 Sold_1[53] + 500 Lost_0[53] + 500 Lost_1[53]
   + 500 Left_1[53] + profits_vars[53] = 0
 Fraction_lower_bound_54: Fraction[54] >= 0
 Fraction_upper_bound_54: Fraction[54] <= 1
 R542: 0.7114940766369874 alphas[0] - 0.701383365180884 alphas[1]
   - alphas[2] + f_var[54] = 0
 R543: - 196.8542016893929 Fraction[54] + Q0_var[54] = 0
 R544: Q0_var[54] + Lost_0_expr_54 = 142.7823488390995
 R545: Q0_var[54] + Q_hat_adjusted_54 = 179.053361344184
 R546: Sold_0[54] - Q0_var[54] - Q1_var[54] + total_stock_second_period_54
   = 0
 R547: total_stock_second_period_54 + Lost_1_expr_54 = 41.10370433131273
 R548: Sold_1[54] + Left_1[54] - total_stock_second_period_54 = 0
 R549: - 500 Sold_0[54] - 500 Sold_1[54] + 500 Lost_0[54] + 500 Lost_1[54]
   + 500 Left_1[54] + profits_vars[54] = 0
 Fraction_lower_bound_55: Fraction[55] >= 0
 Fraction_upper_bound_55: Fraction[55] <= 1
 R552: 1.396557142730432 alphas[0] - 0.8931694350433583 alphas[1]
   - alphas[2] + f_var[55] = 0
 R553: - 196.8542016893929 Fraction[55] + Q0_var[55] = 0
 R554: Q0_var[55] + Lost_0_expr_55 = 167.1392950295579
 R555: Q0_var[55] + Q_hat_adjusted_55 = 195.52855685265
 R556: Sold_0[55] - Q0_var[55] - Q1_var[55] + total_stock_second_period_55
   = 0
 R557: total_stock_second_period_55 + Lost_1_expr_55 = 27.35237077829932
 R558: Sold_1[55] + Left_1[55] - total_stock_second_period_55 = 0
 R559: - 500 Sold_0[55] - 500 Sold_1[55] + 500 Lost_0[55] + 500 Lost_1[55]
   + 500 Left_1[55] + profits_vars[55] = 0
 Fraction_lower_bound_56: Fraction[56] >= 0
 Fraction_upper_bound_56: Fraction[56] <= 1
 R562: - 0.2339301045333848 alphas[0] + 1.407545192179608 alphas[1]
   - alphas[2] + f_var[56] = 0
 R563: - 196.8542016893929 Fraction[56] + Q0_var[56] = 0
 R564: Q0_var[56] + Lost_0_expr_56 = 147.0648716001043
 R565: Q0_var[56] + Q_hat_adjusted_56 = 184.7261320130273
 R566: Sold_0[56] - Q0_var[56] - Q1_var[56] + total_stock_second_period_56
   = 0
 R567: total_stock_second_period_56 + Lost_1_expr_56 = 40.76312124636966
 R568: Sold_1[56] + Left_1[56] - total_stock_second_period_56 = 0
 R569: - 500 Sold_0[56] - 500 Sold_1[56] + 500 Lost_0[56] + 500 Lost_1[56]
   + 500 Left_1[56] + profits_vars[56] = 0
 Fraction_lower_bound_57: Fraction[57] >= 0
 Fraction_upper_bound_57: Fraction[57] <= 1
 R572: - 0.5518642710509637 alphas[0] + 0.8334522330251875 alphas[1]
   - alphas[2] + f_var[57] = 0
 R573: - 196.8542016893929 Fraction[57] + Q0_var[57] = 0
 R574: Q0_var[57] + Lost_0_expr_57 = 170.9339369866948
 R575: Q0_var[57] + Q_hat_adjusted_57 = 204.5508588727715
 R576: Sold_0[57] - Q0_var[57] - Q1_var[57] + total_stock_second_period_57
   = 0
 R577: total_stock_second_period_57 + Lost_1_expr_57 = 30.69793527690408
 R578: Sold_1[57] + Left_1[57] - total_stock_second_period_57 = 0
 R579: - 500 Sold_0[57] - 500 Sold_1[57] + 500 Lost_0[57] + 500 Lost_1[57]
   + 500 Left_1[57] + profits_vars[57] = 0
 Fraction_lower_bound_58: Fraction[58] >= 0
 Fraction_upper_bound_58: Fraction[58] <= 1
 R582: 1.68719949300327 alphas[0] + 1.115745969574116 alphas[1] - alphas[2]
   + f_var[58] = 0
 R583: - 196.8542016893929 Fraction[58] + Q0_var[58] = 0
 R584: Q0_var[58] + Lost_0_expr_58 = 108.4500881062024
 R585: Q0_var[58] + Q_hat_adjusted_58 = 143.4624353378
 R586: Sold_0[58] - Q0_var[58] - Q1_var[58] + total_stock_second_period_58
   = 0
 R587: total_stock_second_period_58 + Lost_1_expr_58 = 38.23896956665394
 R588: Sold_1[58] + Left_1[58] - total_stock_second_period_58 = 0
 R589: - 500 Sold_0[58] - 500 Sold_1[58] + 500 Lost_0[58] + 500 Lost_1[58]
   + 500 Left_1[58] + profits_vars[58] = 0
 Fraction_lower_bound_59: Fraction[59] >= 0
 Fraction_upper_bound_59: Fraction[59] <= 1
 R592: - 0.0689724228015137 alphas[0] - 1.630771534040876 alphas[1]
   - alphas[2] + f_var[59] = 0
 R593: - 196.8542016893929 Fraction[59] + Q0_var[59] = 0
 R594: Q0_var[59] + Lost_0_expr_59 = 169.2242896656427
 R595: Q0_var[59] + Q_hat_adjusted_59 = 199.5577684885131
 R596: Sold_0[59] - Q0_var[59] - Q1_var[59] + total_stock_second_period_59
   = 0
 R597: total_stock_second_period_59 + Lost_1_expr_59 = 24.41022416776264
 R598: Sold_1[59] + Left_1[59] - total_stock_second_period_59 = 0
 R599: - 500 Sold_0[59] - 500 Sold_1[59] + 500 Lost_0[59] + 500 Lost_1[59]
   + 500 Left_1[59] + profits_vars[59] = 0
 Fraction_lower_bound_60: Fraction[60] >= 0
 Fraction_upper_bound_60: Fraction[60] <= 1
 R602: - 1.088164996656323 alphas[0] - 0.608770835777499 alphas[1]
   - alphas[2] + f_var[60] = 0
 R603: - 196.8542016893929 Fraction[60] + Q0_var[60] = 0
 R604: Q0_var[60] + Lost_0_expr_60 = 219.1298882765454
 R605: Q0_var[60] + Q_hat_adjusted_60 = 273.4411960721204
 R606: Sold_0[60] - Q0_var[60] - Q1_var[60] + total_stock_second_period_60
   = 0
 R607: total_stock_second_period_60 + Lost_1_expr_60 = 55.93560242520565
 R608: Sold_1[60] + Left_1[60] - total_stock_second_period_60 = 0
 R609: - 500 Sold_0[60] - 500 Sold_1[60] + 500 Lost_0[60] + 500 Lost_1[60]
   + 500 Left_1[60] + profits_vars[60] = 0
 Fraction_lower_bound_61: Fraction[61] >= 0
 Fraction_upper_bound_61: Fraction[61] <= 1
 R612: - 1.277824293102741 alphas[0] + 1.754611107912161 alphas[1]
   - alphas[2] + f_var[61] = 0
 R613: - 196.8542016893929 Fraction[61] + Q0_var[61] = 0
 R614: Q0_var[61] + Lost_0_expr_61 = 155.7283838942932
 R615: Q0_var[61] + Q_hat_adjusted_61 = 198.2148117062424
 R616: Sold_0[61] - Q0_var[61] - Q1_var[61] + total_stock_second_period_61
   = 0
 R617: total_stock_second_period_61 + Lost_1_expr_61 = 44.09256990594078
 R618: Sold_1[61] + Left_1[61] - total_stock_second_period_61 = 0
 R619: - 500 Sold_0[61] - 500 Sold_1[61] + 500 Lost_0[61] + 500 Lost_1[61]
   + 500 Left_1[61] + profits_vars[61] = 0
 Fraction_lower_bound_62: Fraction[62] >= 0
 Fraction_upper_bound_62: Fraction[62] <= 1
 R622: - 0.9500470411768825 alphas[0] + 1.09406077599894 alphas[1]
   - alphas[2] + f_var[62] = 0
 R623: - 196.8542016893929 Fraction[62] + Q0_var[62] = 0
 R624: Q0_var[62] + Lost_0_expr_62 = 158.5921863149329
 R625: Q0_var[62] + Q_hat_adjusted_62 = 195.1078092320771
 R626: Sold_0[62] - Q0_var[62] - Q1_var[62] + total_stock_second_period_62
   = 0
 R627: total_stock_second_period_62 + Lost_1_expr_62 = 37.41968679443328
 R628: Sold_1[62] + Left_1[62] - total_stock_second_period_62 = 0
 R629: - 500 Sold_0[62] - 500 Sold_1[62] + 500 Lost_0[62] + 500 Lost_1[62]
   + 500 Left_1[62] + profits_vars[62] = 0
 Fraction_lower_bound_63: Fraction[63] >= 0
 Fraction_upper_bound_63: Fraction[63] <= 1
 R632: 1.711900742735368 alphas[0] - 0.8022401204045705 alphas[1]
   - alphas[2] + f_var[63] = 0
 R633: - 196.8542016893929 Fraction[63] + Q0_var[63] = 0
 R634: Q0_var[63] + Lost_0_expr_63 = 111.2370020201212
 R635: Q0_var[63] + Q_hat_adjusted_63 = 149.1517578232805
 R636: Sold_0[63] - Q0_var[63] - Q1_var[63] + total_stock_second_period_63
   = 0
 R637: total_stock_second_period_63 + Lost_1_expr_63 = 39.56453887794157
 R638: Sold_1[63] + Left_1[63] - total_stock_second_period_63 = 0
 R639: - 500 Sold_0[63] - 500 Sold_1[63] + 500 Lost_0[63] + 500 Lost_1[63]
   + 500 Left_1[63] + profits_vars[63] = 0
 Fraction_lower_bound_64: Fraction[64] >= 0
 Fraction_upper_bound_64: Fraction[64] <= 1
 R642: 0.7000000267642009 alphas[0] - 0.696291245517062 alphas[1]
   - alphas[2] + f_var[64] = 0
 R643: - 196.8542016893929 Fraction[64] + Q0_var[64] = 0
 R644: Q0_var[64] + Lost_0_expr_64 = 193.6529752264836
 R645: Q0_var[64] + Q_hat_adjusted_64 = 208.4706620931459
 R646: Sold_0[64] - Q0_var[64] - Q1_var[64] + total_stock_second_period_64
   = 0
 R647: total_stock_second_period_64 + Lost_1_expr_64 = 16.48751770652235
 R648: Sold_1[64] + Left_1[64] - total_stock_second_period_64 = 0
 R649: - 500 Sold_0[64] - 500 Sold_1[64] + 500 Lost_0[64] + 500 Lost_1[64]
   + 500 Left_1[64] + profits_vars[64] = 0
 Fraction_lower_bound_65: Fraction[65] >= 0
 Fraction_upper_bound_65: Fraction[65] <= 1
 R652: 0.5471754628202417 alphas[0] - 1.07033566967403 alphas[1]
   - alphas[2] + f_var[65] = 0
 R653: - 196.8542016893929 Fraction[65] + Q0_var[65] = 0
 R654: Q0_var[65] + Lost_0_expr_65 = 139.1738025176523
 R655: Q0_var[65] + Q_hat_adjusted_65 = 184.0721230831901
 R656: Sold_0[65] - Q0_var[65] - Q1_var[65] + total_stock_second_period_65
   = 0
 R657: total_stock_second_period_65 + Lost_1_expr_65 = 42.36141164802137
 R658: Sold_1[65] + Left_1[65] - total_stock_second_period_65 = 0
 R659: - 500 Sold_0[65] - 500 Sold_1[65] + 500 Lost_0[65] + 500 Lost_1[65]
   + 500 Left_1[65] + profits_vars[65] = 0
 Fraction_lower_bound_66: Fraction[66] >= 0
 Fraction_upper_bound_66: Fraction[66] <= 1
 R662: 0.160378555773121 alphas[0] - 0.0376890127841145 alphas[1]
   - alphas[2] + f_var[66] = 0
 R663: - 196.8542016893929 Fraction[66] + Q0_var[66] = 0
 R664: Q0_var[66] + Lost_0_expr_66 = 174.489332164002
 R665: Q0_var[66] + Q_hat_adjusted_66 = 219.5135039676645
 R666: Sold_0[66] - Q0_var[66] - Q1_var[66] + total_stock_second_period_66
   = 0
 R667: total_stock_second_period_66 + Lost_1_expr_66 = 42.65404858334712
 R668: Sold_1[66] + Left_1[66] - total_stock_second_period_66 = 0
 R669: - 500 Sold_0[66] - 500 Sold_1[66] + 500 Lost_0[66] + 500 Lost_1[66]
   + 500 Left_1[66] + profits_vars[66] = 0
 Fraction_lower_bound_67: Fraction[67] >= 0
 Fraction_upper_bound_67: Fraction[67] <= 1
 R672: 0.2311946944101281 alphas[0] + 0.1788875323880172 alphas[1]
   - alphas[2] + f_var[67] = 0
 R673: - 196.8542016893929 Fraction[67] + Q0_var[67] = 0
 R674: Q0_var[67] + Lost_0_expr_67 = 176.0821461448771
 R675: Q0_var[67] + Q_hat_adjusted_67 = 213.322846318944
 R676: Sold_0[67] - Q0_var[67] - Q1_var[67] + total_stock_second_period_67
   = 0
 R677: total_stock_second_period_67 + Lost_1_expr_67 = 43.25836245904946
 R678: Sold_1[67] + Left_1[67] - total_stock_second_period_67 = 0
 R679: - 500 Sold_0[67] - 500 Sold_1[67] + 500 Lost_0[67] + 500 Lost_1[67]
   + 500 Left_1[67] + profits_vars[67] = 0
 Fraction_lower_bound_68: Fraction[68] >= 0
 Fraction_upper_bound_68: Fraction[68] <= 1
 R682: 1.153637118203759 alphas[0] - 0.6755024148475599 alphas[1]
   - alphas[2] + f_var[68] = 0
 R683: - 196.8542016893929 Fraction[68] + Q0_var[68] = 0
 R684: Q0_var[68] + Lost_0_expr_68 = 187.1039628490162
 R685: Q0_var[68] + Q_hat_adjusted_68 = 203.4171217346931
 R686: Sold_0[68] - Q0_var[68] - Q1_var[68] + total_stock_second_period_68
   = 0
 R687: total_stock_second_period_68 + Lost_1_expr_68 = 11.70977477065042
 R688: Sold_1[68] + Left_1[68] - total_stock_second_period_68 = 0
 R689: - 500 Sold_0[68] - 500 Sold_1[68] + 500 Lost_0[68] + 500 Lost_1[68]
   + 500 Left_1[68] + profits_vars[68] = 0
 Fraction_lower_bound_69: Fraction[69] >= 0
 Fraction_upper_bound_69: Fraction[69] <= 1
 R692: - 1.45710948196159 alphas[0] + 0.7469948882293795 alphas[1]
   - alphas[2] + f_var[69] = 0
 R693: - 196.8542016893929 Fraction[69] + Q0_var[69] = 0
 R694: Q0_var[69] + Lost_0_expr_69 = 178.8597944691199
 R695: Q0_var[69] + Q_hat_adjusted_69 = 224.555487072664
 R696: Sold_0[69] - Q0_var[69] - Q1_var[69] + total_stock_second_period_69
   = 0
 R697: total_stock_second_period_69 + Lost_1_expr_69 = 44.64715767238908
 R698: Sold_1[69] + Left_1[69] - total_stock_second_period_69 = 0
 R699: - 500 Sold_0[69] - 500 Sold_1[69] + 500 Lost_0[69] + 500 Lost_1[69]
   + 500 Left_1[69] + profits_vars[69] = 0
 Fraction_lower_bound_70: Fraction[70] >= 0
 Fraction_upper_bound_70: Fraction[70] <= 1
 R702: 0.3052140891599044 alphas[0] - 0.8270666694259657 alphas[1]
   - alphas[2] + f_var[70] = 0
 R703: - 196.8542016893929 Fraction[70] + Q0_var[70] = 0
 R704: Q0_var[70] + Lost_0_expr_70 = 209.5362450463267
 R705: Q0_var[70] + Q_hat_adjusted_70 = 250.4695344884297
 R706: Sold_0[70] - Q0_var[70] - Q1_var[70] + total_stock_second_period_70
   = 0
 R707: total_stock_second_period_70 + Lost_1_expr_70 = 42.60571415344077
 R708: Sold_1[70] + Left_1[70] - total_stock_second_period_70 = 0
 R709: - 500 Sold_0[70] - 500 Sold_1[70] + 500 Lost_0[70] + 500 Lost_1[70]
   + 500 Left_1[70] + profits_vars[70] = 0
 Fraction_lower_bound_71: Fraction[71] >= 0
 Fraction_upper_bound_71: Fraction[71] <= 1
 R712: 1.736402628146639 alphas[0] - 1.680750276466356 alphas[1]
   - alphas[2] + f_var[71] = 0
 R713: - 196.8542016893929 Fraction[71] + Q0_var[71] = 0
 R714: Q0_var[71] + Lost_0_expr_71 = 127.6877224049384
 R715: Q0_var[71] + Q_hat_adjusted_71 = 166.3311615413495
 R716: Sold_0[71] - Q0_var[71] - Q1_var[71] + total_stock_second_period_71
   = 0
 R717: total_stock_second_period_71 + Lost_1_expr_71 = 36.02121606885284
 R718: Sold_1[71] + Left_1[71] - total_stock_second_period_71 = 0
 R719: - 500 Sold_0[71] - 500 Sold_1[71] + 500 Lost_0[71] + 500 Lost_1[71]
   + 500 Left_1[71] + profits_vars[71] = 0
 Fraction_lower_bound_72: Fraction[72] >= 0
 Fraction_upper_bound_72: Fraction[72] <= 1
 R722: 1.528856261843994 alphas[0] - 0.4076871580255242 alphas[1]
   - alphas[2] + f_var[72] = 0
 R723: - 196.8542016893929 Fraction[72] + Q0_var[72] = 0
 R724: Q0_var[72] + Lost_0_expr_72 = 126.8561896912158
 R725: Q0_var[72] + Q_hat_adjusted_72 = 164.1052905382617
 R726: Sold_0[72] - Q0_var[72] - Q1_var[72] + total_stock_second_period_72
   = 0
 R727: total_stock_second_period_72 + Lost_1_expr_72 = 30.16745871990294
 R728: Sold_1[72] + Left_1[72] - total_stock_second_period_72 = 0
 R729: - 500 Sold_0[72] - 500 Sold_1[72] + 500 Lost_0[72] + 500 Lost_1[72]
   + 500 Left_1[72] + profits_vars[72] = 0
 Fraction_lower_bound_73: Fraction[73] >= 0
 Fraction_upper_bound_73: Fraction[73] <= 1
 R732: 1.33831728866893 alphas[0] - 1.359488163411579 alphas[1] - alphas[2]
   + f_var[73] = 0
 R733: - 196.8542016893929 Fraction[73] + Q0_var[73] = 0
 R734: Q0_var[73] + Lost_0_expr_73 = 134.4590154845495
 R735: Q0_var[73] + Q_hat_adjusted_73 = 164.0089786891079
 R736: Sold_0[73] - Q0_var[73] - Q1_var[73] + total_stock_second_period_73
   = 0
 R737: total_stock_second_period_73 + Lost_1_expr_73 = 27.32089275225794
 R738: Sold_1[73] + Left_1[73] - total_stock_second_period_73 = 0
 R739: - 500 Sold_0[73] - 500 Sold_1[73] + 500 Lost_0[73] + 500 Lost_1[73]
   + 500 Left_1[73] + profits_vars[73] = 0
 Fraction_lower_bound_74: Fraction[74] >= 0
 Fraction_upper_bound_74: Fraction[74] <= 1
 R742: 0.3685971861212176 alphas[0] + 0.9750714587691671 alphas[1]
   - alphas[2] + f_var[74] = 0
 R743: - 196.8542016893929 Fraction[74] + Q0_var[74] = 0
 R744: Q0_var[74] + Lost_0_expr_74 = 149.8727009724957
 R745: Q0_var[74] + Q_hat_adjusted_74 = 196.2336896753856
 R746: Sold_0[74] - Q0_var[74] - Q1_var[74] + total_stock_second_period_74
   = 0
 R747: total_stock_second_period_74 + Lost_1_expr_74 = 42.01056888619795
 R748: Sold_1[74] + Left_1[74] - total_stock_second_period_74 = 0
 R749: - 500 Sold_0[74] - 500 Sold_1[74] + 500 Lost_0[74] + 500 Lost_1[74]
   + 500 Left_1[74] + profits_vars[74] = 0
 Fraction_lower_bound_75: Fraction[75] >= 0
 Fraction_upper_bound_75: Fraction[75] <= 1
 R752: - 0.7406142499320443 alphas[0] - 0.2579679334548292 alphas[1]
   - alphas[2] + f_var[75] = 0
 R753: - 196.8542016893929 Fraction[75] + Q0_var[75] = 0
 R754: Q0_var[75] + Lost_0_expr_75 = 144.4060288169521
 R755: Q0_var[75] + Q_hat_adjusted_75 = 196.5247598233773
 R756: Sold_0[75] - Q0_var[75] - Q1_var[75] + total_stock_second_period_75
   = 0
 R757: total_stock_second_period_75 + Lost_1_expr_75 = 43.85569567206009
 R758: Sold_1[75] + Left_1[75] - total_stock_second_period_75 = 0
 R759: - 500 Sold_0[75] - 500 Sold_1[75] + 500 Lost_0[75] + 500 Lost_1[75]
   + 500 Left_1[75] + profits_vars[75] = 0
 Fraction_lower_bound_76: Fraction[76] >= 0
 Fraction_upper_bound_76: Fraction[76] <= 1
 R762: - 1.543999424523153 alphas[0] - 1.265761200412606 alphas[1]
   - alphas[2] + f_var[76] = 0
 R763: - 196.8542016893929 Fraction[76] + Q0_var[76] = 0
 R764: Q0_var[76] + Lost_0_expr_76 = 187.1649917443912
 R765: Q0_var[76] + Q_hat_adjusted_76 = 239.4600601817824
 R766: Sold_0[76] - Q0_var[76] - Q1_var[76] + total_stock_second_period_76
   = 0
 R767: total_stock_second_period_76 + Lost_1_expr_76 = 54.18363988891223
 R768: Sold_1[76] + Left_1[76] - total_stock_second_period_76 = 0
 R769: - 500 Sold_0[76] - 500 Sold_1[76] + 500 Lost_0[76] + 500 Lost_1[76]
   + 500 Left_1[76] + profits_vars[76] = 0
 Fraction_lower_bound_77: Fraction[77] >= 0
 Fraction_upper_bound_77: Fraction[77] <= 1
 R772: - 1.657068023442862 alphas[0] + 0.6306276607627351 alphas[1]
   - alphas[2] + f_var[77] = 0
 R773: - 196.8542016893929 Fraction[77] + Q0_var[77] = 0
 R774: Q0_var[77] + Lost_0_expr_77 = 200.4817687263207
 R775: Q0_var[77] + Q_hat_adjusted_77 = 246.5978248238824
 R776: Sold_0[77] - Q0_var[77] - Q1_var[77] + total_stock_second_period_77
   = 0
 R777: total_stock_second_period_77 + Lost_1_expr_77 = 41.44687413218951
 R778: Sold_1[77] + Left_1[77] - total_stock_second_period_77 = 0
 R779: - 500 Sold_0[77] - 500 Sold_1[77] + 500 Lost_0[77] + 500 Lost_1[77]
   + 500 Left_1[77] + profits_vars[77] = 0
 Fraction_lower_bound_78: Fraction[78] >= 0
 Fraction_upper_bound_78: Fraction[78] <= 1
 R782: 1.333189576057811 alphas[0] - 1.080770546321778 alphas[1]
   - alphas[2] + f_var[78] = 0
 R783: - 196.8542016893929 Fraction[78] + Q0_var[78] = 0
 R784: Q0_var[78] + Lost_0_expr_78 = 141.0867520977398
 R785: Q0_var[78] + Q_hat_adjusted_78 = 178.6332229477667
 R786: Sold_0[78] - Q0_var[78] - Q1_var[78] + total_stock_second_period_78
   = 0
 R787: total_stock_second_period_78 + Lost_1_expr_78 = 32.05698546626781
 R788: Sold_1[78] + Left_1[78] - total_stock_second_period_78 = 0
 R789: - 500 Sold_0[78] - 500 Sold_1[78] + 500 Lost_0[78] + 500 Lost_1[78]
   + 500 Left_1[78] + profits_vars[78] = 0
 Fraction_lower_bound_79: Fraction[79] >= 0
 Fraction_upper_bound_79: Fraction[79] <= 1
 R792: - 0.4811629537951557 alphas[0] - 1.65089458258736 alphas[1]
   - alphas[2] + f_var[79] = 0
 R793: - 196.8542016893929 Fraction[79] + Q0_var[79] = 0
 R794: Q0_var[79] + Lost_0_expr_79 = 179.9588959570796
 R795: Q0_var[79] + Q_hat_adjusted_79 = 220.2768764936971
 R796: Sold_0[79] - Q0_var[79] - Q1_var[79] + total_stock_second_period_79
   = 0
 R797: total_stock_second_period_79 + Lost_1_expr_79 = 52.00835466278781
 R798: Sold_1[79] + Left_1[79] - total_stock_second_period_79 = 0
 R799: - 500 Sold_0[79] - 500 Sold_1[79] + 500 Lost_0[79] + 500 Lost_1[79]
   + 500 Left_1[79] + profits_vars[79] = 0
 Fraction_lower_bound_80: Fraction[80] >= 0
 Fraction_upper_bound_80: Fraction[80] <= 1
 R802: - 1.234805826960657 alphas[0] + 0.1835346782970739 alphas[1]
   - alphas[2] + f_var[80] = 0
 R803: - 196.8542016893929 Fraction[80] + Q0_var[80] = 0
 R804: Q0_var[80] + Lost_0_expr_80 = 137.909463056904
 R805: Q0_var[80] + Q_hat_adjusted_80 = 179.7326458131745
 R806: Sold_0[80] - Q0_var[80] - Q1_var[80] + total_stock_second_period_80
   = 0
 R807: total_stock_second_period_80 + Lost_1_expr_80 = 35.55882625385622
 R808: Sold_1[80] + Left_1[80] - total_stock_second_period_80 = 0
 R809: - 500 Sold_0[80] - 500 Sold_1[80] + 500 Lost_0[80] + 500 Lost_1[80]
   + 500 Left_1[80] + profits_vars[80] = 0
 Fraction_lower_bound_81: Fraction[81] >= 0
 Fraction_upper_bound_81: Fraction[81] <= 1
 R812: 0.3066068035478883 alphas[0] - 1.26540947597776 alphas[1]
   - alphas[2] + f_var[81] = 0
 R813: - 196.8542016893929 Fraction[81] + Q0_var[81] = 0
 R814: Q0_var[81] + Lost_0_expr_81 = 194.7953867082214
 R815: Q0_var[81] + Q_hat_adjusted_81 = 230.3027285199519
 R816: Sold_0[81] - Q0_var[81] - Q1_var[81] + total_stock_second_period_81
   = 0
 R817: total_stock_second_period_81 + Lost_1_expr_81 = 32.65147107403249
 R818: Sold_1[81] + Left_1[81] - total_stock_second_period_81 = 0
 R819: - 500 Sold_0[81] - 500 Sold_1[81] + 500 Lost_0[81] + 500 Lost_1[81]
   + 500 Left_1[81] + profits_vars[81] = 0
 Fraction_lower_bound_82: Fraction[82] >= 0
 Fraction_upper_bound_82: Fraction[82] <= 1
 R822: - 0.7960537083450505 alphas[0] - 0.4485999845964631 alphas[1]
   - alphas[2] + f_var[82] = 0
 R823: - 196.8542016893929 Fraction[82] + Q0_var[82] = 0
 R824: Q0_var[82] + Lost_0_expr_82 = 185.6621554781706
 R825: Q0_var[82] + Q_hat_adjusted_82 = 219.9648918675977
 R826: Sold_0[82] - Q0_var[82] - Q1_var[82] + total_stock_second_period_82
   = 0
 R827: total_stock_second_period_82 + Lost_1_expr_82 = 35.76264088228669
 R828: Sold_1[82] + Left_1[82] - total_stock_second_period_82 = 0
 R829: - 500 Sold_0[82] - 500 Sold_1[82] + 500 Lost_0[82] + 500 Lost_1[82]
   + 500 Left_1[82] + profits_vars[82] = 0
 Fraction_lower_bound_83: Fraction[83] >= 0
 Fraction_upper_bound_83: Fraction[83] <= 1
 R832: - 1.017995705542013 alphas[0] + 0.9949193360384968 alphas[1]
   - alphas[2] + f_var[83] = 0
 R833: - 196.8542016893929 Fraction[83] + Q0_var[83] = 0
 R834: Q0_var[83] + Lost_0_expr_83 = 193.8708587716901
 R835: Q0_var[83] + Q_hat_adjusted_83 = 251.5057812566789
 R836: Sold_0[83] - Q0_var[83] - Q1_var[83] + total_stock_second_period_83
   = 0
 R837: total_stock_second_period_83 + Lost_1_expr_83 = 51.53131182691543
 R838: Sold_1[83] + Left_1[83] - total_stock_second_period_83 = 0
 R839: - 500 Sold_0[83] - 500 Sold_1[83] + 500 Lost_0[83] + 500 Lost_1[83]
   + 500 Left_1[83] + profits_vars[83] = 0
 Fraction_lower_bound_84: Fraction[84] >= 0
 Fraction_upper_bound_84: Fraction[84] <= 1
 R842: 0.3626090175734729 alphas[0] + 1.653085740467708 alphas[1]
   - alphas[2] + f_var[84] = 0
 R843: - 196.8542016893929 Fraction[84] + Q0_var[84] = 0
 R844: Q0_var[84] + Lost_0_expr_84 = 144.8937012503487
 R845: Q0_var[84] + Q_hat_adjusted_84 = 185.956092332078
 R846: Sold_0[84] - Q0_var[84] - Q1_var[84] + total_stock_second_period_84
   = 0
 R847: total_stock_second_period_84 + Lost_1_expr_84 = 38.20676985333871
 R848: Sold_1[84] + Left_1[84] - total_stock_second_period_84 = 0
 R849: - 500 Sold_0[84] - 500 Sold_1[84] + 500 Lost_0[84] + 500 Lost_1[84]
   + 500 Left_1[84] + profits_vars[84] = 0
 Fraction_lower_bound_85: Fraction[85] >= 0
 Fraction_upper_bound_85: Fraction[85] <= 1
 R852: - 0.6274328842407945 alphas[0] - 0.292272086274906 alphas[1]
   - alphas[2] + f_var[85] = 0
 R853: - 196.8542016893929 Fraction[85] + Q0_var[85] = 0
 R854: Q0_var[85] + Lost_0_expr_85 = 164.6139178493565
 R855: Q0_var[85] + Q_hat_adjusted_85 = 218.0216997005842
 R856: Sold_0[85] - Q0_var[85] - Q1_var[85] + total_stock_second_period_85
   = 0
 R857: total_stock_second_period_85 + Lost_1_expr_85 = 48.83425220580024
 R858: Sold_1[85] + Left_1[85] - total_stock_second_period_85 = 0
 R859: - 500 Sold_0[85] - 500 Sold_1[85] + 500 Lost_0[85] + 500 Lost_1[85]
   + 500 Left_1[85] + profits_vars[85] = 0
 Fraction_lower_bound_86: Fraction[86] >= 0
 Fraction_upper_bound_86: Fraction[86] <= 1
 R862: 0.1249085662038234 alphas[0] - 1.684839873016819 alphas[1]
   - alphas[2] + f_var[86] = 0
 R863: - 196.8542016893929 Fraction[86] + Q0_var[86] = 0
 R864: Q0_var[86] + Lost_0_expr_86 = 168.8824193031614
 R865: Q0_var[86] + Q_hat_adjusted_86 = 204.4159572205924
 R866: Sold_0[86] - Q0_var[86] - Q1_var[86] + total_stock_second_period_86
   = 0
 R867: total_stock_second_period_86 + Lost_1_expr_86 = 37.49119311317673
 R868: Sold_1[86] + Left_1[86] - total_stock_second_period_86 = 0
 R869: - 500 Sold_0[86] - 500 Sold_1[86] + 500 Lost_0[86] + 500 Lost_1[86]
   + 500 Left_1[86] + profits_vars[86] = 0
 Fraction_lower_bound_87: Fraction[87] >= 0
 Fraction_upper_bound_87: Fraction[87] <= 1
 R872: - 1.377040088881901 alphas[0] + 0.509267799178169 alphas[1]
   - alphas[2] + f_var[87] = 0
 R873: - 196.8542016893929 Fraction[87] + Q0_var[87] = 0
 R874: Q0_var[87] + Lost_0_expr_87 = 159.4601654528038
 R875: Q0_var[87] + Q_hat_adjusted_87 = 189.555984848361
 R876: Sold_0[87] - Q0_var[87] - Q1_var[87] + total_stock_second_period_87
   = 0
 R877: total_stock_second_period_87 + Lost_1_expr_87 = 21.02506297062278
 R878: Sold_1[87] + Left_1[87] - total_stock_second_period_87 = 0
 R879: - 500 Sold_0[87] - 500 Sold_1[87] + 500 Lost_0[87] + 500 Lost_1[87]
   + 500 Left_1[87] + profits_vars[87] = 0
 Fraction_lower_bound_88: Fraction[88] >= 0
 Fraction_upper_bound_88: Fraction[88] <= 1
 R882: - 0.2578285687753638 alphas[0] + 0.0395554796104062 alphas[1]
   - alphas[2] + f_var[88] = 0
 R883: - 196.8542016893929 Fraction[88] + Q0_var[88] = 0
 R884: Q0_var[88] + Lost_0_expr_88 = 147.5755304769682
 R885: Q0_var[88] + Q_hat_adjusted_88 = 191.3180258932695
 R886: Sold_0[88] - Q0_var[88] - Q1_var[88] + total_stock_second_period_88
   = 0
 R887: total_stock_second_period_88 + Lost_1_expr_88 = 44.74473720311897
 R888: Sold_1[88] + Left_1[88] - total_stock_second_period_88 = 0
 R889: - 500 Sold_0[88] - 500 Sold_1[88] + 500 Lost_0[88] + 500 Lost_1[88]
   + 500 Left_1[88] + profits_vars[88] = 0
 Fraction_lower_bound_89: Fraction[89] >= 0
 Fraction_upper_bound_89: Fraction[89] <= 1
 R892: - 1.656015060159153 alphas[0] - 1.742594137399191 alphas[1]
   - alphas[2] + f_var[89] = 0
 R893: - 196.8542016893929 Fraction[89] + Q0_var[89] = 0
 R894: Q0_var[89] + Lost_0_expr_89 = 210.6818402308549
 R895: Q0_var[89] + Q_hat_adjusted_89 = 260.7154760237598
 R896: Sold_0[89] - Q0_var[89] - Q1_var[89] + total_stock_second_period_89
   = 0
 R897: total_stock_second_period_89 + Lost_1_expr_89 = 48.67744150491736
 R898: Sold_1[89] + Left_1[89] - total_stock_second_period_89 = 0
 R899: - 500 Sold_0[89] - 500 Sold_1[89] + 500 Lost_0[89] + 500 Lost_1[89]
   + 500 Left_1[89] + profits_vars[89] = 0
 Fraction_lower_bound_90: Fraction[90] >= 0
 Fraction_upper_bound_90: Fraction[90] <= 1
 R902: 1.424849167288132 alphas[0] + 0.1101618232087178 alphas[1]
   - alphas[2] + f_var[90] = 0
 R903: - 196.8542016893929 Fraction[90] + Q0_var[90] = 0
 R904: Q0_var[90] + Lost_0_expr_90 = 127.1126558029728
 R905: Q0_var[90] + Q_hat_adjusted_90 = 163.899698039337
 R906: Sold_0[90] - Q0_var[90] - Q1_var[90] + total_stock_second_period_90
   = 0
 R907: total_stock_second_period_90 + Lost_1_expr_90 = 44.85579831361785
 R908: Sold_1[90] + Left_1[90] - total_stock_second_period_90 = 0
 R909: - 500 Sold_0[90] - 500 Sold_1[90] + 500 Lost_0[90] + 500 Lost_1[90]
   + 500 Left_1[90] + profits_vars[90] = 0
 Fraction_lower_bound_91: Fraction[91] >= 0
 Fraction_upper_bound_91: Fraction[91] <= 1
 R912: - 0.3840251340608806 alphas[0] + 0.4969543924249765 alphas[1]
   - alphas[2] + f_var[91] = 0
 R913: - 196.8542016893929 Fraction[91] + Q0_var[91] = 0
 R914: Q0_var[91] + Lost_0_expr_91 = 171.802098043029
 R915: Q0_var[91] + Q_hat_adjusted_91 = 207.5864343457139
 R916: Sold_0[91] - Q0_var[91] - Q1_var[91] + total_stock_second_period_91
   = 0
 R917: total_stock_second_period_91 + Lost_1_expr_91 = 34.00201459909109
 R918: Sold_1[91] + Left_1[91] - total_stock_second_period_91 = 0
 R919: - 500 Sold_0[91] - 500 Sold_1[91] + 500 Lost_0[91] + 500 Lost_1[91]
   + 500 Left_1[91] + profits_vars[91] = 0
 Fraction_lower_bound_92: Fraction[92] >= 0
 Fraction_upper_bound_92: Fraction[92] <= 1
 R922: - 1.037059512377409 alphas[0] - 1.353410449291161 alphas[1]
   - alphas[2] + f_var[92] = 0
 R923: - 196.8542016893929 Fraction[92] + Q0_var[92] = 0
 R924: Q0_var[92] + Lost_0_expr_92 = 198.7380501048704
 R925: Q0_var[92] + Q_hat_adjusted_92 = 249.1463083545214
 R926: Sold_0[92] - Q0_var[92] - Q1_var[92] + total_stock_second_period_92
   = 0
 R927: total_stock_second_period_92 + Lost_1_expr_92 = 42.68992497194125
 R928: Sold_1[92] + Left_1[92] - total_stock_second_period_92 = 0
 R929: - 500 Sold_0[92] - 500 Sold_1[92] + 500 Lost_0[92] + 500 Lost_1[92]
   + 500 Left_1[92] + profits_vars[92] = 0
 Fraction_lower_bound_93: Fraction[93] >= 0
 Fraction_upper_bound_93: Fraction[93] <= 1
 R932: 0.4142042454263009 alphas[0] - 1.442575276528452 alphas[1]
   - alphas[2] + f_var[93] = 0
 R933: - 196.8542016893929 Fraction[93] + Q0_var[93] = 0
 R934: Q0_var[93] + Lost_0_expr_93 = 168.0181837344112
 R935: Q0_var[93] + Q_hat_adjusted_93 = 192.7650946815489
 R936: Sold_0[93] - Q0_var[93] - Q1_var[93] + total_stock_second_period_93
   = 0
 R937: total_stock_second_period_93 + Lost_1_expr_93 = 17.81611197634654
 R938: Sold_1[93] + Left_1[93] - total_stock_second_period_93 = 0
 R939: - 500 Sold_0[93] - 500 Sold_1[93] + 500 Lost_0[93] + 500 Lost_1[93]
   + 500 Left_1[93] + profits_vars[93] = 0
 Fraction_lower_bound_94: Fraction[94] >= 0
 Fraction_upper_bound_94: Fraction[94] <= 1
 R942: - 1.606195431483485 alphas[0] + 1.584827689950216 alphas[1]
   - alphas[2] + f_var[94] = 0
 R943: - 196.8542016893929 Fraction[94] + Q0_var[94] = 0
 R944: Q0_var[94] + Lost_0_expr_94 = 182.0244600588858
 R945: Q0_var[94] + Q_hat_adjusted_94 = 228.7186986844619
 R946: Sold_0[94] - Q0_var[94] - Q1_var[94] + total_stock_second_period_94
   = 0
 R947: total_stock_second_period_94 + Lost_1_expr_94 = 47.1720090033094
 R948: Sold_1[94] + Left_1[94] - total_stock_second_period_94 = 0
 R949: - 500 Sold_0[94] - 500 Sold_1[94] + 500 Lost_0[94] + 500 Lost_1[94]
   + 500 Left_1[94] + profits_vars[94] = 0
 Fraction_lower_bound_95: Fraction[95] >= 0
 Fraction_upper_bound_95: Fraction[95] <= 1
 R952: 1.413814970368536 alphas[0] - 0.2908831256711207 alphas[1]
   - alphas[2] + f_var[95] = 0
 R953: - 196.8542016893929 Fraction[95] + Q0_var[95] = 0
 R954: Q0_var[95] + Lost_0_expr_95 = 104.775557376525
 R955: Q0_var[95] + Q_hat_adjusted_95 = 156.2961357135429
 R956: Sold_0[95] - Q0_var[95] - Q1_var[95] + total_stock_second_period_95
   = 0
 R957: total_stock_second_period_95 + Lost_1_expr_95 = 52.18225684358342
 R958: Sold_1[95] + Left_1[95] - total_stock_second_period_95 = 0
 R959: - 500 Sold_0[95] - 500 Sold_1[95] + 500 Lost_0[95] + 500 Lost_1[95]
   + 500 Left_1[95] + profits_vars[95] = 0
 Fraction_lower_bound_96: Fraction[96] >= 0
 Fraction_upper_bound_96: Fraction[96] <= 1
 R962: 1.102796693804723 alphas[0] + 0.9256391632102229 alphas[1]
   - alphas[2] + f_var[96] = 0
 R963: - 196.8542016893929 Fraction[96] + Q0_var[96] = 0
 R964: Q0_var[96] + Lost_0_expr_96 = 123.4956486879957
 R965: Q0_var[96] + Q_hat_adjusted_96 = 160.75124850854
 R966: Sold_0[96] - Q0_var[96] - Q1_var[96] + total_stock_second_period_96
   = 0
 R967: total_stock_second_period_96 + Lost_1_expr_96 = 47.14933388031224
 R968: Sold_1[96] + Left_1[96] - total_stock_second_period_96 = 0
 R969: - 500 Sold_0[96] - 500 Sold_1[96] + 500 Lost_0[96] + 500 Lost_1[96]
   + 500 Left_1[96] + profits_vars[96] = 0
 Fraction_lower_bound_97: Fraction[97] >= 0
 Fraction_upper_bound_97: Fraction[97] <= 1
 R972: - 0.2764228941749043 alphas[0] - 0.183081275602807 alphas[1]
   - alphas[2] + f_var[97] = 0
 R973: - 196.8542016893929 Fraction[97] + Q0_var[97] = 0
 R974: Q0_var[97] + Lost_0_expr_97 = 165.5940273035478
 R975: Q0_var[97] + Q_hat_adjusted_97 = 211.9293727143412
 R976: Sold_0[97] - Q0_var[97] - Q1_var[97] + total_stock_second_period_97
   = 0
 R977: total_stock_second_period_97 + Lost_1_expr_97 = 50.11806578038843
 R978: Sold_1[97] + Left_1[97] - total_stock_second_period_97 = 0
 R979: - 500 Sold_0[97] - 500 Sold_1[97] + 500 Lost_0[97] + 500 Lost_1[97]
   + 500 Left_1[97] + profits_vars[97] = 0
 Fraction_lower_bound_98: Fraction[98] >= 0
 Fraction_upper_bound_98: Fraction[98] <= 1
 R982: 1.302714159537228 alphas[0] + 1.205540359633948 alphas[1]
   - alphas[2] + f_var[98] = 0
 R983: - 196.8542016893929 Fraction[98] + Q0_var[98] = 0
 R984: Q0_var[98] + Lost_0_expr_98 = 124.4483663042755
 R985: Q0_var[98] + Q_hat_adjusted_98 = 155.6347931668162
 R986: Sold_0[98] - Q0_var[98] - Q1_var[98] + total_stock_second_period_98
   = 0
 R987: total_stock_second_period_98 + Lost_1_expr_98 = 34.28650994071167
 R988: Sold_1[98] + Left_1[98] - total_stock_second_period_98 = 0
 R989: - 500 Sold_0[98] - 500 Sold_1[98] + 500 Lost_0[98] + 500 Lost_1[98]
   + 500 Left_1[98] + profits_vars[98] = 0
 Fraction_lower_bound_99: Fraction[99] >= 0
 Fraction_upper_bound_99: Fraction[99] <= 1
 R992: 1.054608756600154 alphas[0] - 0.9232924744875651 alphas[1]
   - alphas[2] + f_var[99] = 0
 R993: - 196.8542016893929 Fraction[99] + Q0_var[99] = 0
 R994: Q0_var[99] + Lost_0_expr_99 = 184.6207118571966
 R995: Q0_var[99] + Q_hat_adjusted_99 = 226.7436143697271
 R996: Sold_0[99] - Q0_var[99] - Q1_var[99] + total_stock_second_period_99
   = 0
 R997: total_stock_second_period_99 + Lost_1_expr_99 = 45.84421530674086
 R998: Sold_1[99] + Left_1[99] - total_stock_second_period_99 = 0
 R999: - 500 Sold_0[99] - 500 Sold_1[99] + 500 Lost_0[99] + 500 Lost_1[99]
   + 500 Left_1[99] + profits_vars[99] = 0
Bounds
 -10 <= alphas[0] <= 10
 -10 <= alphas[1] <= 10
 -10 <= alphas[2] <= 10
 -10 <= f_var[0] <= 10
 -10 <= f_var[1] <= 10
 -10 <= f_var[2] <= 10
 -10 <= f_var[3] <= 10
 -10 <= f_var[4] <= 10
 -10 <= f_var[5] <= 10
 -10 <= f_var[6] <= 10
 -10 <= f_var[7] <= 10
 -10 <= f_var[8] <= 10
 -10 <= f_var[9] <= 10
 -10 <= f_var[10] <= 10
 -10 <= f_var[11] <= 10
 -10 <= f_var[12] <= 10
 -10 <= f_var[13] <= 10
 -10 <= f_var[14] <= 10
 -10 <= f_var[15] <= 10
 -10 <= f_var[16] <= 10
 -10 <= f_var[17] <= 10
 -10 <= f_var[18] <= 10
 -10 <= f_var[19] <= 10
 -10 <= f_var[20] <= 10
 -10 <= f_var[21] <= 10
 -10 <= f_var[22] <= 10
 -10 <= f_var[23] <= 10
 -10 <= f_var[24] <= 10
 -10 <= f_var[25] <= 10
 -10 <= f_var[26] <= 10
 -10 <= f_var[27] <= 10
 -10 <= f_var[28] <= 10
 -10 <= f_var[29] <= 10
 -10 <= f_var[30] <= 10
 -10 <= f_var[31] <= 10
 -10 <= f_var[32] <= 10
 -10 <= f_var[33] <= 10
 -10 <= f_var[34] <= 10
 -10 <= f_var[35] <= 10
 -10 <= f_var[36] <= 10
 -10 <= f_var[37] <= 10
 -10 <= f_var[38] <= 10
 -10 <= f_var[39] <= 10
 -10 <= f_var[40] <= 10
 -10 <= f_var[41] <= 10
 -10 <= f_var[42] <= 10
 -10 <= f_var[43] <= 10
 -10 <= f_var[44] <= 10
 -10 <= f_var[45] <= 10
 -10 <= f_var[46] <= 10
 -10 <= f_var[47] <= 10
 -10 <= f_var[48] <= 10
 -10 <= f_var[49] <= 10
 -10 <= f_var[50] <= 10
 -10 <= f_var[51] <= 10
 -10 <= f_var[52] <= 10
 -10 <= f_var[53] <= 10
 -10 <= f_var[54] <= 10
 -10 <= f_var[55] <= 10
 -10 <= f_var[56] <= 10
 -10 <= f_var[57] <= 10
 -10 <= f_var[58] <= 10
 -10 <= f_var[59] <= 10
 -10 <= f_var[60] <= 10
 -10 <= f_var[61] <= 10
 -10 <= f_var[62] <= 10
 -10 <= f_var[63] <= 10
 -10 <= f_var[64] <= 10
 -10 <= f_var[65] <= 10
 -10 <= f_var[66] <= 10
 -10 <= f_var[67] <= 10
 -10 <= f_var[68] <= 10
 -10 <= f_var[69] <= 10
 -10 <= f_var[70] <= 10
 -10 <= f_var[71] <= 10
 -10 <= f_var[72] <= 10
 -10 <= f_var[73] <= 10
 -10 <= f_var[74] <= 10
 -10 <= f_var[75] <= 10
 -10 <= f_var[76] <= 10
 -10 <= f_var[77] <= 10
 -10 <= f_var[78] <= 10
 -10 <= f_var[79] <= 10
 -10 <= f_var[80] <= 10
 -10 <= f_var[81] <= 10
 -10 <= f_var[82] <= 10
 -10 <= f_var[83] <= 10
 -10 <= f_var[84] <= 10
 -10 <= f_var[85] <= 10
 -10 <= f_var[86] <= 10
 -10 <= f_var[87] <= 10
 -10 <= f_var[88] <= 10
 -10 <= f_var[89] <= 10
 -10 <= f_var[90] <= 10
 -10 <= f_var[91] <= 10
 -10 <= f_var[92] <= 10
 -10 <= f_var[93] <= 10
 -10 <= f_var[94] <= 10
 -10 <= f_var[95] <= 10
 -10 <= f_var[96] <= 10
 -10 <= f_var[97] <= 10
 -10 <= f_var[98] <= 10
 -10 <= f_var[99] <= 10
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
 Q0_var[0] <= 197.8542016893929
 Q0_var[1] <= 197.8542016893929
 Q0_var[2] <= 197.8542016893929
 Q0_var[3] <= 197.8542016893929
 Q0_var[4] <= 197.8542016893929
 Q0_var[5] <= 197.8542016893929
 Q0_var[6] <= 197.8542016893929
 Q0_var[7] <= 197.8542016893929
 Q0_var[8] <= 197.8542016893929
 Q0_var[9] <= 197.8542016893929
 Q0_var[10] <= 197.8542016893929
 Q0_var[11] <= 197.8542016893929
 Q0_var[12] <= 197.8542016893929
 Q0_var[13] <= 197.8542016893929
 Q0_var[14] <= 197.8542016893929
 Q0_var[15] <= 197.8542016893929
 Q0_var[16] <= 197.8542016893929
 Q0_var[17] <= 197.8542016893929
 Q0_var[18] <= 197.8542016893929
 Q0_var[19] <= 197.8542016893929
 Q0_var[20] <= 197.8542016893929
 Q0_var[21] <= 197.8542016893929
 Q0_var[22] <= 197.8542016893929
 Q0_var[23] <= 197.8542016893929
 Q0_var[24] <= 197.8542016893929
 Q0_var[25] <= 197.8542016893929
 Q0_var[26] <= 197.8542016893929
 Q0_var[27] <= 197.8542016893929
 Q0_var[28] <= 197.8542016893929
 Q0_var[29] <= 197.8542016893929
 Q0_var[30] <= 197.8542016893929
 Q0_var[31] <= 197.8542016893929
 Q0_var[32] <= 197.8542016893929
 Q0_var[33] <= 197.8542016893929
 Q0_var[34] <= 197.8542016893929
 Q0_var[35] <= 197.8542016893929
 Q0_var[36] <= 197.8542016893929
 Q0_var[37] <= 197.8542016893929
 Q0_var[38] <= 197.8542016893929
 Q0_var[39] <= 197.8542016893929
 Q0_var[40] <= 197.8542016893929
 Q0_var[41] <= 197.8542016893929
 Q0_var[42] <= 197.8542016893929
 Q0_var[43] <= 197.8542016893929
 Q0_var[44] <= 197.8542016893929
 Q0_var[45] <= 197.8542016893929
 Q0_var[46] <= 197.8542016893929
 Q0_var[47] <= 197.8542016893929
 Q0_var[48] <= 197.8542016893929
 Q0_var[49] <= 197.8542016893929
 Q0_var[50] <= 197.8542016893929
 Q0_var[51] <= 197.8542016893929
 Q0_var[52] <= 197.8542016893929
 Q0_var[53] <= 197.8542016893929
 Q0_var[54] <= 197.8542016893929
 Q0_var[55] <= 197.8542016893929
 Q0_var[56] <= 197.8542016893929
 Q0_var[57] <= 197.8542016893929
 Q0_var[58] <= 197.8542016893929
 Q0_var[59] <= 197.8542016893929
 Q0_var[60] <= 197.8542016893929
 Q0_var[61] <= 197.8542016893929
 Q0_var[62] <= 197.8542016893929
 Q0_var[63] <= 197.8542016893929
 Q0_var[64] <= 197.8542016893929
 Q0_var[65] <= 197.8542016893929
 Q0_var[66] <= 197.8542016893929
 Q0_var[67] <= 197.8542016893929
 Q0_var[68] <= 197.8542016893929
 Q0_var[69] <= 197.8542016893929
 Q0_var[70] <= 197.8542016893929
 Q0_var[71] <= 197.8542016893929
 Q0_var[72] <= 197.8542016893929
 Q0_var[73] <= 197.8542016893929
 Q0_var[74] <= 197.8542016893929
 Q0_var[75] <= 197.8542016893929
 Q0_var[76] <= 197.8542016893929
 Q0_var[77] <= 197.8542016893929
 Q0_var[78] <= 197.8542016893929
 Q0_var[79] <= 197.8542016893929
 Q0_var[80] <= 197.8542016893929
 Q0_var[81] <= 197.8542016893929
 Q0_var[82] <= 197.8542016893929
 Q0_var[83] <= 197.8542016893929
 Q0_var[84] <= 197.8542016893929
 Q0_var[85] <= 197.8542016893929
 Q0_var[86] <= 197.8542016893929
 Q0_var[87] <= 197.8542016893929
 Q0_var[88] <= 197.8542016893929
 Q0_var[89] <= 197.8542016893929
 Q0_var[90] <= 197.8542016893929
 Q0_var[91] <= 197.8542016893929
 Q0_var[92] <= 197.8542016893929
 Q0_var[93] <= 197.8542016893929
 Q0_var[94] <= 197.8542016893929
 Q0_var[95] <= 197.8542016893929
 Q0_var[96] <= 197.8542016893929
 Q0_var[97] <= 197.8542016893929
 Q0_var[98] <= 197.8542016893929
 Q0_var[99] <= 197.8542016893929
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
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 171.9705526854737 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 40.00056744336146 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 124.9174642130688 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 56.96688789215834 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 178.7053681962325 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 35.96797259406476 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 137.3831500547491 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 21.56865094348434 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 143.9535066724153 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 6.943726725029627 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 199.2437125823047 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 35.59143750066534 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 159.6723925421878 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 25.57223421387051 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 171.2283386759027 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 17.69412214989524 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 132.0321983449813 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 22.07722047943105 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 150.9931699477943 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 20.74203343463805 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 109.7827951535821 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 25.86372640942696 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 156.9217355903971 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 31.05309535586031 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 110.2675355014723 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 22.23244107479613 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 133.5567173791324 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 43.48610893436775 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 181.912974906696 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 37.40275210109417 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 140.5069759576123 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 22.1281266149289 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 193.5398129351872 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 38.77726993647558 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 163.2283874502225 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 47.98179273765295 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 181.0505986101853 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 4.477984455359557 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 204.9390731849088 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 39.81206344818219 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 137.1546967403893 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 53.64953356637277 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 153.1190901099322 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 38.21243883440913 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 132.4254014425122 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 28.18128674784869 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 171.474319846935 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 56.63239264835586 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 170.1947067262712 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 34.19186156250329 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 171.2695199646037 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 33.48575890842393 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 148.0716584518483 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 47.71958188970855 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 197.746021833302 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 27.85371514600896 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 214.6649974676589 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 43.80366275174308 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 106.0300190055475 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 27.66386612557105 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 160.9523400246461 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 46.90148567368223 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 170.2989269211271 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 14.90864950417645 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 158.755917130324 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 23.64679122005043 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 145.6860643563938 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 25.53716001953845 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 198.5514384897298 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 41.5614366355288 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 109.7540047812708 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 32.22749803008396 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 135.5815287331823 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 39.65949512585815 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 187.5707800870579 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 58.15954461047021 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 164.5066144839163 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 39.14235574889312 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 120.5427379818348 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 20.55161373790484 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 179.6732770887298 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 36.88608347891664 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 153.8877293941184 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 45.95059525813855 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 197.4094493607441 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 47.99268709071016 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 130.0818736708006 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 47.66134363513352 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 159.2351471373275 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 58.84888756857456 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 187.8292467518336 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 27.92334495927061 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 188.1533080893215 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 35.33877421277447 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 159.8675131459254 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 37.24957815626567 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 128.7438596425112 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 36.6554026870744 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 145.7571018959608 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 41.38627351476885 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
 GC300: Fraction[50] = LOGISTIC ( f_var[50] )
 min_sales_constr_50: Sold_0[50] = MIN ( Q0_var[50] , 112.1953665086659 )
 max_lost_constr_50: Lost_0[50] = MAX ( Lost_0_expr_50 , 0 )
 max_Q1_constr_50: Q1_var[50] = MAX ( Q_hat_adjusted_50 , 0 )
 min_sales2_constr_50: Sold_1[50] = MIN ( total_stock_second_period_50
   , 33.58445448056638 )
 max_lost2_constr_50: Lost_1[50] = MAX ( Lost_1_expr_50 , 0 )
 GC306: Fraction[51] = LOGISTIC ( f_var[51] )
 min_sales_constr_51: Sold_0[51] = MIN ( Q0_var[51] , 158.4857945539278 )
 max_lost_constr_51: Lost_0[51] = MAX ( Lost_0_expr_51 , 0 )
 max_Q1_constr_51: Q1_var[51] = MAX ( Q_hat_adjusted_51 , 0 )
 min_sales2_constr_51: Sold_1[51] = MIN ( total_stock_second_period_51
   , 8.350276965676997 )
 max_lost2_constr_51: Lost_1[51] = MAX ( Lost_1_expr_51 , 0 )
 GC312: Fraction[52] = LOGISTIC ( f_var[52] )
 min_sales_constr_52: Sold_0[52] = MIN ( Q0_var[52] , 195.4216836397702 )
 max_lost_constr_52: Lost_0[52] = MAX ( Lost_0_expr_52 , 0 )
 max_Q1_constr_52: Q1_var[52] = MAX ( Q_hat_adjusted_52 , 0 )
 min_sales2_constr_52: Sold_1[52] = MIN ( total_stock_second_period_52
   , 42.40169722335882 )
 max_lost2_constr_52: Lost_1[52] = MAX ( Lost_1_expr_52 , 0 )
 GC318: Fraction[53] = LOGISTIC ( f_var[53] )
 min_sales_constr_53: Sold_0[53] = MIN ( Q0_var[53] , 162.6858902368283 )
 max_lost_constr_53: Lost_0[53] = MAX ( Lost_0_expr_53 , 0 )
 max_Q1_constr_53: Q1_var[53] = MAX ( Q_hat_adjusted_53 , 0 )
 min_sales2_constr_53: Sold_1[53] = MIN ( total_stock_second_period_53
   , 42.47178348039976 )
 max_lost2_constr_53: Lost_1[53] = MAX ( Lost_1_expr_53 , 0 )
 GC324: Fraction[54] = LOGISTIC ( f_var[54] )
 min_sales_constr_54: Sold_0[54] = MIN ( Q0_var[54] , 142.7823488390995 )
 max_lost_constr_54: Lost_0[54] = MAX ( Lost_0_expr_54 , 0 )
 max_Q1_constr_54: Q1_var[54] = MAX ( Q_hat_adjusted_54 , 0 )
 min_sales2_constr_54: Sold_1[54] = MIN ( total_stock_second_period_54
   , 41.10370433131273 )
 max_lost2_constr_54: Lost_1[54] = MAX ( Lost_1_expr_54 , 0 )
 GC330: Fraction[55] = LOGISTIC ( f_var[55] )
 min_sales_constr_55: Sold_0[55] = MIN ( Q0_var[55] , 167.1392950295579 )
 max_lost_constr_55: Lost_0[55] = MAX ( Lost_0_expr_55 , 0 )
 max_Q1_constr_55: Q1_var[55] = MAX ( Q_hat_adjusted_55 , 0 )
 min_sales2_constr_55: Sold_1[55] = MIN ( total_stock_second_period_55
   , 27.35237077829932 )
 max_lost2_constr_55: Lost_1[55] = MAX ( Lost_1_expr_55 , 0 )
 GC336: Fraction[56] = LOGISTIC ( f_var[56] )
 min_sales_constr_56: Sold_0[56] = MIN ( Q0_var[56] , 147.0648716001043 )
 max_lost_constr_56: Lost_0[56] = MAX ( Lost_0_expr_56 , 0 )
 max_Q1_constr_56: Q1_var[56] = MAX ( Q_hat_adjusted_56 , 0 )
 min_sales2_constr_56: Sold_1[56] = MIN ( total_stock_second_period_56
   , 40.76312124636966 )
 max_lost2_constr_56: Lost_1[56] = MAX ( Lost_1_expr_56 , 0 )
 GC342: Fraction[57] = LOGISTIC ( f_var[57] )
 min_sales_constr_57: Sold_0[57] = MIN ( Q0_var[57] , 170.9339369866948 )
 max_lost_constr_57: Lost_0[57] = MAX ( Lost_0_expr_57 , 0 )
 max_Q1_constr_57: Q1_var[57] = MAX ( Q_hat_adjusted_57 , 0 )
 min_sales2_constr_57: Sold_1[57] = MIN ( total_stock_second_period_57
   , 30.69793527690408 )
 max_lost2_constr_57: Lost_1[57] = MAX ( Lost_1_expr_57 , 0 )
 GC348: Fraction[58] = LOGISTIC ( f_var[58] )
 min_sales_constr_58: Sold_0[58] = MIN ( Q0_var[58] , 108.4500881062024 )
 max_lost_constr_58: Lost_0[58] = MAX ( Lost_0_expr_58 , 0 )
 max_Q1_constr_58: Q1_var[58] = MAX ( Q_hat_adjusted_58 , 0 )
 min_sales2_constr_58: Sold_1[58] = MIN ( total_stock_second_period_58
   , 38.23896956665394 )
 max_lost2_constr_58: Lost_1[58] = MAX ( Lost_1_expr_58 , 0 )
 GC354: Fraction[59] = LOGISTIC ( f_var[59] )
 min_sales_constr_59: Sold_0[59] = MIN ( Q0_var[59] , 169.2242896656427 )
 max_lost_constr_59: Lost_0[59] = MAX ( Lost_0_expr_59 , 0 )
 max_Q1_constr_59: Q1_var[59] = MAX ( Q_hat_adjusted_59 , 0 )
 min_sales2_constr_59: Sold_1[59] = MIN ( total_stock_second_period_59
   , 24.41022416776264 )
 max_lost2_constr_59: Lost_1[59] = MAX ( Lost_1_expr_59 , 0 )
 GC360: Fraction[60] = LOGISTIC ( f_var[60] )
 min_sales_constr_60: Sold_0[60] = MIN ( Q0_var[60] , 219.1298882765454 )
 max_lost_constr_60: Lost_0[60] = MAX ( Lost_0_expr_60 , 0 )
 max_Q1_constr_60: Q1_var[60] = MAX ( Q_hat_adjusted_60 , 0 )
 min_sales2_constr_60: Sold_1[60] = MIN ( total_stock_second_period_60
   , 55.93560242520565 )
 max_lost2_constr_60: Lost_1[60] = MAX ( Lost_1_expr_60 , 0 )
 GC366: Fraction[61] = LOGISTIC ( f_var[61] )
 min_sales_constr_61: Sold_0[61] = MIN ( Q0_var[61] , 155.7283838942932 )
 max_lost_constr_61: Lost_0[61] = MAX ( Lost_0_expr_61 , 0 )
 max_Q1_constr_61: Q1_var[61] = MAX ( Q_hat_adjusted_61 , 0 )
 min_sales2_constr_61: Sold_1[61] = MIN ( total_stock_second_period_61
   , 44.09256990594078 )
 max_lost2_constr_61: Lost_1[61] = MAX ( Lost_1_expr_61 , 0 )
 GC372: Fraction[62] = LOGISTIC ( f_var[62] )
 min_sales_constr_62: Sold_0[62] = MIN ( Q0_var[62] , 158.5921863149329 )
 max_lost_constr_62: Lost_0[62] = MAX ( Lost_0_expr_62 , 0 )
 max_Q1_constr_62: Q1_var[62] = MAX ( Q_hat_adjusted_62 , 0 )
 min_sales2_constr_62: Sold_1[62] = MIN ( total_stock_second_period_62
   , 37.41968679443328 )
 max_lost2_constr_62: Lost_1[62] = MAX ( Lost_1_expr_62 , 0 )
 GC378: Fraction[63] = LOGISTIC ( f_var[63] )
 min_sales_constr_63: Sold_0[63] = MIN ( Q0_var[63] , 111.2370020201212 )
 max_lost_constr_63: Lost_0[63] = MAX ( Lost_0_expr_63 , 0 )
 max_Q1_constr_63: Q1_var[63] = MAX ( Q_hat_adjusted_63 , 0 )
 min_sales2_constr_63: Sold_1[63] = MIN ( total_stock_second_period_63
   , 39.56453887794157 )
 max_lost2_constr_63: Lost_1[63] = MAX ( Lost_1_expr_63 , 0 )
 GC384: Fraction[64] = LOGISTIC ( f_var[64] )
 min_sales_constr_64: Sold_0[64] = MIN ( Q0_var[64] , 193.6529752264836 )
 max_lost_constr_64: Lost_0[64] = MAX ( Lost_0_expr_64 , 0 )
 max_Q1_constr_64: Q1_var[64] = MAX ( Q_hat_adjusted_64 , 0 )
 min_sales2_constr_64: Sold_1[64] = MIN ( total_stock_second_period_64
   , 16.48751770652235 )
 max_lost2_constr_64: Lost_1[64] = MAX ( Lost_1_expr_64 , 0 )
 GC390: Fraction[65] = LOGISTIC ( f_var[65] )
 min_sales_constr_65: Sold_0[65] = MIN ( Q0_var[65] , 139.1738025176523 )
 max_lost_constr_65: Lost_0[65] = MAX ( Lost_0_expr_65 , 0 )
 max_Q1_constr_65: Q1_var[65] = MAX ( Q_hat_adjusted_65 , 0 )
 min_sales2_constr_65: Sold_1[65] = MIN ( total_stock_second_period_65
   , 42.36141164802137 )
 max_lost2_constr_65: Lost_1[65] = MAX ( Lost_1_expr_65 , 0 )
 GC396: Fraction[66] = LOGISTIC ( f_var[66] )
 min_sales_constr_66: Sold_0[66] = MIN ( Q0_var[66] , 174.489332164002 )
 max_lost_constr_66: Lost_0[66] = MAX ( Lost_0_expr_66 , 0 )
 max_Q1_constr_66: Q1_var[66] = MAX ( Q_hat_adjusted_66 , 0 )
 min_sales2_constr_66: Sold_1[66] = MIN ( total_stock_second_period_66
   , 42.65404858334712 )
 max_lost2_constr_66: Lost_1[66] = MAX ( Lost_1_expr_66 , 0 )
 GC402: Fraction[67] = LOGISTIC ( f_var[67] )
 min_sales_constr_67: Sold_0[67] = MIN ( Q0_var[67] , 176.0821461448771 )
 max_lost_constr_67: Lost_0[67] = MAX ( Lost_0_expr_67 , 0 )
 max_Q1_constr_67: Q1_var[67] = MAX ( Q_hat_adjusted_67 , 0 )
 min_sales2_constr_67: Sold_1[67] = MIN ( total_stock_second_period_67
   , 43.25836245904946 )
 max_lost2_constr_67: Lost_1[67] = MAX ( Lost_1_expr_67 , 0 )
 GC408: Fraction[68] = LOGISTIC ( f_var[68] )
 min_sales_constr_68: Sold_0[68] = MIN ( Q0_var[68] , 187.1039628490162 )
 max_lost_constr_68: Lost_0[68] = MAX ( Lost_0_expr_68 , 0 )
 max_Q1_constr_68: Q1_var[68] = MAX ( Q_hat_adjusted_68 , 0 )
 min_sales2_constr_68: Sold_1[68] = MIN ( total_stock_second_period_68
   , 11.70977477065042 )
 max_lost2_constr_68: Lost_1[68] = MAX ( Lost_1_expr_68 , 0 )
 GC414: Fraction[69] = LOGISTIC ( f_var[69] )
 min_sales_constr_69: Sold_0[69] = MIN ( Q0_var[69] , 178.8597944691199 )
 max_lost_constr_69: Lost_0[69] = MAX ( Lost_0_expr_69 , 0 )
 max_Q1_constr_69: Q1_var[69] = MAX ( Q_hat_adjusted_69 , 0 )
 min_sales2_constr_69: Sold_1[69] = MIN ( total_stock_second_period_69
   , 44.64715767238908 )
 max_lost2_constr_69: Lost_1[69] = MAX ( Lost_1_expr_69 , 0 )
 GC420: Fraction[70] = LOGISTIC ( f_var[70] )
 min_sales_constr_70: Sold_0[70] = MIN ( Q0_var[70] , 209.5362450463267 )
 max_lost_constr_70: Lost_0[70] = MAX ( Lost_0_expr_70 , 0 )
 max_Q1_constr_70: Q1_var[70] = MAX ( Q_hat_adjusted_70 , 0 )
 min_sales2_constr_70: Sold_1[70] = MIN ( total_stock_second_period_70
   , 42.60571415344077 )
 max_lost2_constr_70: Lost_1[70] = MAX ( Lost_1_expr_70 , 0 )
 GC426: Fraction[71] = LOGISTIC ( f_var[71] )
 min_sales_constr_71: Sold_0[71] = MIN ( Q0_var[71] , 127.6877224049384 )
 max_lost_constr_71: Lost_0[71] = MAX ( Lost_0_expr_71 , 0 )
 max_Q1_constr_71: Q1_var[71] = MAX ( Q_hat_adjusted_71 , 0 )
 min_sales2_constr_71: Sold_1[71] = MIN ( total_stock_second_period_71
   , 36.02121606885284 )
 max_lost2_constr_71: Lost_1[71] = MAX ( Lost_1_expr_71 , 0 )
 GC432: Fraction[72] = LOGISTIC ( f_var[72] )
 min_sales_constr_72: Sold_0[72] = MIN ( Q0_var[72] , 126.8561896912158 )
 max_lost_constr_72: Lost_0[72] = MAX ( Lost_0_expr_72 , 0 )
 max_Q1_constr_72: Q1_var[72] = MAX ( Q_hat_adjusted_72 , 0 )
 min_sales2_constr_72: Sold_1[72] = MIN ( total_stock_second_period_72
   , 30.16745871990294 )
 max_lost2_constr_72: Lost_1[72] = MAX ( Lost_1_expr_72 , 0 )
 GC438: Fraction[73] = LOGISTIC ( f_var[73] )
 min_sales_constr_73: Sold_0[73] = MIN ( Q0_var[73] , 134.4590154845495 )
 max_lost_constr_73: Lost_0[73] = MAX ( Lost_0_expr_73 , 0 )
 max_Q1_constr_73: Q1_var[73] = MAX ( Q_hat_adjusted_73 , 0 )
 min_sales2_constr_73: Sold_1[73] = MIN ( total_stock_second_period_73
   , 27.32089275225794 )
 max_lost2_constr_73: Lost_1[73] = MAX ( Lost_1_expr_73 , 0 )
 GC444: Fraction[74] = LOGISTIC ( f_var[74] )
 min_sales_constr_74: Sold_0[74] = MIN ( Q0_var[74] , 149.8727009724957 )
 max_lost_constr_74: Lost_0[74] = MAX ( Lost_0_expr_74 , 0 )
 max_Q1_constr_74: Q1_var[74] = MAX ( Q_hat_adjusted_74 , 0 )
 min_sales2_constr_74: Sold_1[74] = MIN ( total_stock_second_period_74
   , 42.01056888619795 )
 max_lost2_constr_74: Lost_1[74] = MAX ( Lost_1_expr_74 , 0 )
 GC450: Fraction[75] = LOGISTIC ( f_var[75] )
 min_sales_constr_75: Sold_0[75] = MIN ( Q0_var[75] , 144.4060288169521 )
 max_lost_constr_75: Lost_0[75] = MAX ( Lost_0_expr_75 , 0 )
 max_Q1_constr_75: Q1_var[75] = MAX ( Q_hat_adjusted_75 , 0 )
 min_sales2_constr_75: Sold_1[75] = MIN ( total_stock_second_period_75
   , 43.85569567206009 )
 max_lost2_constr_75: Lost_1[75] = MAX ( Lost_1_expr_75 , 0 )
 GC456: Fraction[76] = LOGISTIC ( f_var[76] )
 min_sales_constr_76: Sold_0[76] = MIN ( Q0_var[76] , 187.1649917443912 )
 max_lost_constr_76: Lost_0[76] = MAX ( Lost_0_expr_76 , 0 )
 max_Q1_constr_76: Q1_var[76] = MAX ( Q_hat_adjusted_76 , 0 )
 min_sales2_constr_76: Sold_1[76] = MIN ( total_stock_second_period_76
   , 54.18363988891223 )
 max_lost2_constr_76: Lost_1[76] = MAX ( Lost_1_expr_76 , 0 )
 GC462: Fraction[77] = LOGISTIC ( f_var[77] )
 min_sales_constr_77: Sold_0[77] = MIN ( Q0_var[77] , 200.4817687263207 )
 max_lost_constr_77: Lost_0[77] = MAX ( Lost_0_expr_77 , 0 )
 max_Q1_constr_77: Q1_var[77] = MAX ( Q_hat_adjusted_77 , 0 )
 min_sales2_constr_77: Sold_1[77] = MIN ( total_stock_second_period_77
   , 41.44687413218951 )
 max_lost2_constr_77: Lost_1[77] = MAX ( Lost_1_expr_77 , 0 )
 GC468: Fraction[78] = LOGISTIC ( f_var[78] )
 min_sales_constr_78: Sold_0[78] = MIN ( Q0_var[78] , 141.0867520977398 )
 max_lost_constr_78: Lost_0[78] = MAX ( Lost_0_expr_78 , 0 )
 max_Q1_constr_78: Q1_var[78] = MAX ( Q_hat_adjusted_78 , 0 )
 min_sales2_constr_78: Sold_1[78] = MIN ( total_stock_second_period_78
   , 32.05698546626781 )
 max_lost2_constr_78: Lost_1[78] = MAX ( Lost_1_expr_78 , 0 )
 GC474: Fraction[79] = LOGISTIC ( f_var[79] )
 min_sales_constr_79: Sold_0[79] = MIN ( Q0_var[79] , 179.9588959570796 )
 max_lost_constr_79: Lost_0[79] = MAX ( Lost_0_expr_79 , 0 )
 max_Q1_constr_79: Q1_var[79] = MAX ( Q_hat_adjusted_79 , 0 )
 min_sales2_constr_79: Sold_1[79] = MIN ( total_stock_second_period_79
   , 52.00835466278781 )
 max_lost2_constr_79: Lost_1[79] = MAX ( Lost_1_expr_79 , 0 )
 GC480: Fraction[80] = LOGISTIC ( f_var[80] )
 min_sales_constr_80: Sold_0[80] = MIN ( Q0_var[80] , 137.909463056904 )
 max_lost_constr_80: Lost_0[80] = MAX ( Lost_0_expr_80 , 0 )
 max_Q1_constr_80: Q1_var[80] = MAX ( Q_hat_adjusted_80 , 0 )
 min_sales2_constr_80: Sold_1[80] = MIN ( total_stock_second_period_80
   , 35.55882625385622 )
 max_lost2_constr_80: Lost_1[80] = MAX ( Lost_1_expr_80 , 0 )
 GC486: Fraction[81] = LOGISTIC ( f_var[81] )
 min_sales_constr_81: Sold_0[81] = MIN ( Q0_var[81] , 194.7953867082214 )
 max_lost_constr_81: Lost_0[81] = MAX ( Lost_0_expr_81 , 0 )
 max_Q1_constr_81: Q1_var[81] = MAX ( Q_hat_adjusted_81 , 0 )
 min_sales2_constr_81: Sold_1[81] = MIN ( total_stock_second_period_81
   , 32.65147107403249 )
 max_lost2_constr_81: Lost_1[81] = MAX ( Lost_1_expr_81 , 0 )
 GC492: Fraction[82] = LOGISTIC ( f_var[82] )
 min_sales_constr_82: Sold_0[82] = MIN ( Q0_var[82] , 185.6621554781706 )
 max_lost_constr_82: Lost_0[82] = MAX ( Lost_0_expr_82 , 0 )
 max_Q1_constr_82: Q1_var[82] = MAX ( Q_hat_adjusted_82 , 0 )
 min_sales2_constr_82: Sold_1[82] = MIN ( total_stock_second_period_82
   , 35.76264088228669 )
 max_lost2_constr_82: Lost_1[82] = MAX ( Lost_1_expr_82 , 0 )
 GC498: Fraction[83] = LOGISTIC ( f_var[83] )
 min_sales_constr_83: Sold_0[83] = MIN ( Q0_var[83] , 193.8708587716901 )
 max_lost_constr_83: Lost_0[83] = MAX ( Lost_0_expr_83 , 0 )
 max_Q1_constr_83: Q1_var[83] = MAX ( Q_hat_adjusted_83 , 0 )
 min_sales2_constr_83: Sold_1[83] = MIN ( total_stock_second_period_83
   , 51.53131182691543 )
 max_lost2_constr_83: Lost_1[83] = MAX ( Lost_1_expr_83 , 0 )
 GC504: Fraction[84] = LOGISTIC ( f_var[84] )
 min_sales_constr_84: Sold_0[84] = MIN ( Q0_var[84] , 144.8937012503487 )
 max_lost_constr_84: Lost_0[84] = MAX ( Lost_0_expr_84 , 0 )
 max_Q1_constr_84: Q1_var[84] = MAX ( Q_hat_adjusted_84 , 0 )
 min_sales2_constr_84: Sold_1[84] = MIN ( total_stock_second_period_84
   , 38.20676985333871 )
 max_lost2_constr_84: Lost_1[84] = MAX ( Lost_1_expr_84 , 0 )
 GC510: Fraction[85] = LOGISTIC ( f_var[85] )
 min_sales_constr_85: Sold_0[85] = MIN ( Q0_var[85] , 164.6139178493565 )
 max_lost_constr_85: Lost_0[85] = MAX ( Lost_0_expr_85 , 0 )
 max_Q1_constr_85: Q1_var[85] = MAX ( Q_hat_adjusted_85 , 0 )
 min_sales2_constr_85: Sold_1[85] = MIN ( total_stock_second_period_85
   , 48.83425220580024 )
 max_lost2_constr_85: Lost_1[85] = MAX ( Lost_1_expr_85 , 0 )
 GC516: Fraction[86] = LOGISTIC ( f_var[86] )
 min_sales_constr_86: Sold_0[86] = MIN ( Q0_var[86] , 168.8824193031614 )
 max_lost_constr_86: Lost_0[86] = MAX ( Lost_0_expr_86 , 0 )
 max_Q1_constr_86: Q1_var[86] = MAX ( Q_hat_adjusted_86 , 0 )
 min_sales2_constr_86: Sold_1[86] = MIN ( total_stock_second_period_86
   , 37.49119311317673 )
 max_lost2_constr_86: Lost_1[86] = MAX ( Lost_1_expr_86 , 0 )
 GC522: Fraction[87] = LOGISTIC ( f_var[87] )
 min_sales_constr_87: Sold_0[87] = MIN ( Q0_var[87] , 159.4601654528038 )
 max_lost_constr_87: Lost_0[87] = MAX ( Lost_0_expr_87 , 0 )
 max_Q1_constr_87: Q1_var[87] = MAX ( Q_hat_adjusted_87 , 0 )
 min_sales2_constr_87: Sold_1[87] = MIN ( total_stock_second_period_87
   , 21.02506297062278 )
 max_lost2_constr_87: Lost_1[87] = MAX ( Lost_1_expr_87 , 0 )
 GC528: Fraction[88] = LOGISTIC ( f_var[88] )
 min_sales_constr_88: Sold_0[88] = MIN ( Q0_var[88] , 147.5755304769682 )
 max_lost_constr_88: Lost_0[88] = MAX ( Lost_0_expr_88 , 0 )
 max_Q1_constr_88: Q1_var[88] = MAX ( Q_hat_adjusted_88 , 0 )
 min_sales2_constr_88: Sold_1[88] = MIN ( total_stock_second_period_88
   , 44.74473720311897 )
 max_lost2_constr_88: Lost_1[88] = MAX ( Lost_1_expr_88 , 0 )
 GC534: Fraction[89] = LOGISTIC ( f_var[89] )
 min_sales_constr_89: Sold_0[89] = MIN ( Q0_var[89] , 210.6818402308549 )
 max_lost_constr_89: Lost_0[89] = MAX ( Lost_0_expr_89 , 0 )
 max_Q1_constr_89: Q1_var[89] = MAX ( Q_hat_adjusted_89 , 0 )
 min_sales2_constr_89: Sold_1[89] = MIN ( total_stock_second_period_89
   , 48.67744150491736 )
 max_lost2_constr_89: Lost_1[89] = MAX ( Lost_1_expr_89 , 0 )
 GC540: Fraction[90] = LOGISTIC ( f_var[90] )
 min_sales_constr_90: Sold_0[90] = MIN ( Q0_var[90] , 127.1126558029728 )
 max_lost_constr_90: Lost_0[90] = MAX ( Lost_0_expr_90 , 0 )
 max_Q1_constr_90: Q1_var[90] = MAX ( Q_hat_adjusted_90 , 0 )
 min_sales2_constr_90: Sold_1[90] = MIN ( total_stock_second_period_90
   , 44.85579831361785 )
 max_lost2_constr_90: Lost_1[90] = MAX ( Lost_1_expr_90 , 0 )
 GC546: Fraction[91] = LOGISTIC ( f_var[91] )
 min_sales_constr_91: Sold_0[91] = MIN ( Q0_var[91] , 171.802098043029 )
 max_lost_constr_91: Lost_0[91] = MAX ( Lost_0_expr_91 , 0 )
 max_Q1_constr_91: Q1_var[91] = MAX ( Q_hat_adjusted_91 , 0 )
 min_sales2_constr_91: Sold_1[91] = MIN ( total_stock_second_period_91
   , 34.00201459909109 )
 max_lost2_constr_91: Lost_1[91] = MAX ( Lost_1_expr_91 , 0 )
 GC552: Fraction[92] = LOGISTIC ( f_var[92] )
 min_sales_constr_92: Sold_0[92] = MIN ( Q0_var[92] , 198.7380501048704 )
 max_lost_constr_92: Lost_0[92] = MAX ( Lost_0_expr_92 , 0 )
 max_Q1_constr_92: Q1_var[92] = MAX ( Q_hat_adjusted_92 , 0 )
 min_sales2_constr_92: Sold_1[92] = MIN ( total_stock_second_period_92
   , 42.68992497194125 )
 max_lost2_constr_92: Lost_1[92] = MAX ( Lost_1_expr_92 , 0 )
 GC558: Fraction[93] = LOGISTIC ( f_var[93] )
 min_sales_constr_93: Sold_0[93] = MIN ( Q0_var[93] , 168.0181837344112 )
 max_lost_constr_93: Lost_0[93] = MAX ( Lost_0_expr_93 , 0 )
 max_Q1_constr_93: Q1_var[93] = MAX ( Q_hat_adjusted_93 , 0 )
 min_sales2_constr_93: Sold_1[93] = MIN ( total_stock_second_period_93
   , 17.81611197634654 )
 max_lost2_constr_93: Lost_1[93] = MAX ( Lost_1_expr_93 , 0 )
 GC564: Fraction[94] = LOGISTIC ( f_var[94] )
 min_sales_constr_94: Sold_0[94] = MIN ( Q0_var[94] , 182.0244600588858 )
 max_lost_constr_94: Lost_0[94] = MAX ( Lost_0_expr_94 , 0 )
 max_Q1_constr_94: Q1_var[94] = MAX ( Q_hat_adjusted_94 , 0 )
 min_sales2_constr_94: Sold_1[94] = MIN ( total_stock_second_period_94
   , 47.1720090033094 )
 max_lost2_constr_94: Lost_1[94] = MAX ( Lost_1_expr_94 , 0 )
 GC570: Fraction[95] = LOGISTIC ( f_var[95] )
 min_sales_constr_95: Sold_0[95] = MIN ( Q0_var[95] , 104.775557376525 )
 max_lost_constr_95: Lost_0[95] = MAX ( Lost_0_expr_95 , 0 )
 max_Q1_constr_95: Q1_var[95] = MAX ( Q_hat_adjusted_95 , 0 )
 min_sales2_constr_95: Sold_1[95] = MIN ( total_stock_second_period_95
   , 52.18225684358342 )
 max_lost2_constr_95: Lost_1[95] = MAX ( Lost_1_expr_95 , 0 )
 GC576: Fraction[96] = LOGISTIC ( f_var[96] )
 min_sales_constr_96: Sold_0[96] = MIN ( Q0_var[96] , 123.4956486879957 )
 max_lost_constr_96: Lost_0[96] = MAX ( Lost_0_expr_96 , 0 )
 max_Q1_constr_96: Q1_var[96] = MAX ( Q_hat_adjusted_96 , 0 )
 min_sales2_constr_96: Sold_1[96] = MIN ( total_stock_second_period_96
   , 47.14933388031224 )
 max_lost2_constr_96: Lost_1[96] = MAX ( Lost_1_expr_96 , 0 )
 GC582: Fraction[97] = LOGISTIC ( f_var[97] )
 min_sales_constr_97: Sold_0[97] = MIN ( Q0_var[97] , 165.5940273035478 )
 max_lost_constr_97: Lost_0[97] = MAX ( Lost_0_expr_97 , 0 )
 max_Q1_constr_97: Q1_var[97] = MAX ( Q_hat_adjusted_97 , 0 )
 min_sales2_constr_97: Sold_1[97] = MIN ( total_stock_second_period_97
   , 50.11806578038843 )
 max_lost2_constr_97: Lost_1[97] = MAX ( Lost_1_expr_97 , 0 )
 GC588: Fraction[98] = LOGISTIC ( f_var[98] )
 min_sales_constr_98: Sold_0[98] = MIN ( Q0_var[98] , 124.4483663042755 )
 max_lost_constr_98: Lost_0[98] = MAX ( Lost_0_expr_98 , 0 )
 max_Q1_constr_98: Q1_var[98] = MAX ( Q_hat_adjusted_98 , 0 )
 min_sales2_constr_98: Sold_1[98] = MIN ( total_stock_second_period_98
   , 34.28650994071167 )
 max_lost2_constr_98: Lost_1[98] = MAX ( Lost_1_expr_98 , 0 )
 GC594: Fraction[99] = LOGISTIC ( f_var[99] )
 min_sales_constr_99: Sold_0[99] = MIN ( Q0_var[99] , 184.6207118571966 )
 max_lost_constr_99: Lost_0[99] = MAX ( Lost_0_expr_99 , 0 )
 max_Q1_constr_99: Q1_var[99] = MAX ( Q_hat_adjusted_99 , 0 )
 min_sales2_constr_99: Sold_1[99] = MIN ( total_stock_second_period_99
   , 45.84421530674086 )
 max_lost2_constr_99: Lost_1[99] = MAX ( Lost_1_expr_99 , 0 )
End
