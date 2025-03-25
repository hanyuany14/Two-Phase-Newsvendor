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
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: - 0.0398448308177865 alphas[0] + 0.5812771326415103 alphas[1]
   - 0.3672437009088935 alphas[2] + 0.5141232887412315 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 2126.80588765262 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 1477.404818848541
 R5: Q0_var[0] + Q_hat_adjusted_0 = 1875.751524160886
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 396.4401353905788
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.6509235944307136 alphas[0] + 0.3584532793018148 alphas[1]
   + 1.584611044717512 alphas[2] - 0.9327978358556458 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2126.80588765262 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 325.4521829879686
 R15: Q0_var[1] + Q_hat_adjusted_1 = 432.5630045143878
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 105.0318478275103
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.2379962975181053 alphas[0] - 0.1818929165189752 alphas[1]
   + 0.0247708048318948 alphas[2] + 1.348350052169791 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 2126.80588765262 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1171.127177238832
 R25: Q0_var[2] + Q_hat_adjusted_2 = 1510.977617144369
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 337.9130979286114
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.0254092292859822 alphas[0] - 0.9843078950010535 alphas[1]
   - 0.7553601180796957 alphas[2] + 0.3392753955305137 alphas[3]
   - alphas[4] + f_var[3] = 0
 R33: - 2126.80588765262 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 1891.966029414279
 R35: Q0_var[3] + Q_hat_adjusted_3 = 2348.22510652169
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 454.3851954705243
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.4198482119787716 alphas[0] + 0.4990291695375618 alphas[1]
   + 1.471040521181692 alphas[2] - 1.276280578873321 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2126.80588765262 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 978.3055855823229
 R45: Q0_var[4] + Q_hat_adjusted_4 = 1102.422534976048
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 121.9877160107273
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.3964103670188668 alphas[0] - 1.601572854533488 alphas[1]
   - 1.327206121826167 alphas[2] - 0.3032390272298222 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2126.80588765262 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 2174.401044502069
 R55: Q0_var[5] + Q_hat_adjusted_5 = 2716.087720575626
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 539.760741511865
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.3686761193000995 alphas[0] - 1.381643277358852 alphas[1]
   - 0.1234400841199625 alphas[2] - 0.7885075379900311 alphas[3]
   - alphas[4] + f_var[6] = 0
 R63: - 2126.80588765262 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 1528.607093475993
 R65: Q0_var[6] + Q_hat_adjusted_6 = 1890.563125925818
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 360.0407083178366
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.299494290945913 alphas[0] + 1.511957647922241 alphas[1]
   + 1.51563043868907 alphas[2] + 0.1277284572602724 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2126.80588765262 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 322.1606186615509
 R75: Q0_var[7] + Q_hat_adjusted_7 = 439.5753755858569
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 115.3305241628722
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.563536831972461 alphas[0] + 0.7830621643078852 alphas[1]
   + 0.866752426512993 alphas[2] - 1.464833107106158 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2126.80588765262 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 711.607186804144
 R85: Q0_var[8] + Q_hat_adjusted_8 = 925.7861114890763
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 212.2068059604201
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.567546810655553 alphas[0] + 0.0984769268087596 alphas[1]
   - 1.575548841504168 alphas[2] - 0.0119519749267211 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2126.80588765262 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 2228.36595003303
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2807.083004359461
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 576.8378482136565
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: - 0.9320300178845095 alphas[0] + 1.313020270788065 alphas[1]
   - 0.3318764231706242 alphas[2] + 2.00215167773756 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 2126.80588765262 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 1960.353364754076
 R105: Q0_var[10] + Q_hat_adjusted_10 = 2353.450358005676
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 391.1598665260277
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 0.0333137608874336 alphas[0] + 0.4457948666996462 alphas[1]
   + 1.258388391803922 alphas[2] + 1.436757641610959 alphas[3] - alphas[4]
   + f_var[11] = 0
 R113: - 2126.80588765262 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 894.1038050098194
 R115: Q0_var[11] + Q_hat_adjusted_11 = 1049.929900682193
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 153.7362838843955
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: - 0.1104782175581155 alphas[0] - 0.9469744224956345 alphas[1]
   + 0.6553268542087041 alphas[2] + 0.669939894593527 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 2126.80588765262 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 1052.059162567585
 R125: Q0_var[12] + Q_hat_adjusted_12 = 1297.815200199928
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 243.7722511402829
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 1.423724490489904 alphas[0] - 0.7350723008065281 alphas[1]
   + 0.0888123585380466 alphas[2] - 0.0587307399460728 alphas[3]
   - alphas[4] + f_var[13] = 0
 R133: - 2126.80588765262 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 1413.402374621745
 R135: Q0_var[13] + Q_hat_adjusted_13 = 1743.701790510425
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 328.3518129302421
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: 1.714974928057006 alphas[0] - 0.5283376234584164 alphas[1]
   + 1.548152541470626 alphas[2] - 0.6915454658966823 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 2126.80588765262 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 487.9585269049063
 R145: Q0_var[14] + Q_hat_adjusted_14 = 600.4505541358211
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 110.4750346366098
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 1.655866365215913 alphas[0] - 0.841814306337893 alphas[1]
   + 0.2398126039260955 alphas[2] + 0.6391156544128745 alphas[3]
   - alphas[4] + f_var[15] = 0
 R153: - 2126.80588765262 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 1236.303307498596
 R155: Q0_var[15] + Q_hat_adjusted_15 = 1544.061944325097
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 305.8077568068794
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 1.901622154698307 alphas[0] + 0.9563266404768191 alphas[1]
   - 0.0208842829254341 alphas[2] - 0.4621108283837091 alphas[3]
   - alphas[4] + f_var[16] = 0
 R163: - 2126.80588765262 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1174.194375792902
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1520.856062417245
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 344.7293177173491
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 1.082231784035022 alphas[0] + 0.4698998101371899 alphas[1]
   + 0.8022363412370905 alphas[2] - 0.7580651673628089 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 2126.80588765262 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 751.4523840897484
 R175: Q0_var[17] + Q_hat_adjusted_17 = 975.253254011505
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 221.8389376406162
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.8821953122750865 alphas[0] - 0.7198792936112365 alphas[1]
   - 0.3530396712276901 alphas[2] + 0.9944555070814526 alphas[3]
   - alphas[4] + f_var[18] = 0
 R183: - 2126.80588765262 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 1664.311079507421
 R185: Q0_var[18] + Q_hat_adjusted_18 = 2060.520981781607
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 394.3195001459933
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.219569266564456 alphas[0] - 1.441804387526272 alphas[1]
   - 0.9042893946463365 alphas[2] - 1.460397311702637 alphas[3] - alphas[4]
   + f_var[19] = 0
 R193: - 2126.80588765262 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 1824.739408660224
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2303.267135403259
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 476.6200597960884
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 1.618466905718923 alphas[0] - 0.2484249747440853 alphas[1]
   + 0.8285047564187304 alphas[2] - 0.2381300072261339 alphas[3]
   - alphas[4] + f_var[20] = 0
 R203: - 2126.80588765262 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 698.8991175480789
 R205: Q0_var[20] + Q_hat_adjusted_20 = 918.8005134471044
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 217.917112052819
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: - 0.959332475660104 alphas[0] + 0.6797229199004278 alphas[1]
   - 0.4597419365641963 alphas[2] - 1.095986319953991 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 2126.80588765262 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1492.765652653201
 R215: Q0_var[21] + Q_hat_adjusted_21 = 1904.907836505629
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 410.2499501399939
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.2809230890672019 alphas[0] + 1.577591449217005 alphas[1]
   - 0.5237827289983682 alphas[2] + 0.1093962556124357 alphas[3]
   - alphas[4] + f_var[22] = 0
 R223: - 2126.80588765262 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 1823.211351936723
 R225: Q0_var[22] + Q_hat_adjusted_22 = 2244.947998627125
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 419.8111214818708
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 0.890908948497018 alphas[0] - 0.8411329903016457 alphas[1]
   - 1.558350384713815 alphas[2] - 0.5790846527040096 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2126.80588765262 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 2446.584764739198
 R235: Q0_var[23] + Q_hat_adjusted_23 = 3022.60805357297
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 574.2701505644205
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 1.54147401940782 alphas[0] + 1.657081463565447 alphas[1]
   - 0.9881352058482852 alphas[2] - 0.6864029395163715 alphas[3]
   - alphas[4] + f_var[24] = 0
 R243: - 2126.80588765262 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 1989.077137065633
 R245: Q0_var[24] + Q_hat_adjusted_24 = 2480.085590399939
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 489.1380836482219
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: - 0.3744719111560164 alphas[0] + 0.3597186311617322 alphas[1]
   - 1.222796874821057 alphas[2] - 1.197035423037729 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 2126.80588765262 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 2080.988378515591
 R255: Q0_var[25] + Q_hat_adjusted_25 = 2606.913223391509
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 524.1726348950031
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: 1.449362342404136 alphas[0] - 0.2016906773338462 alphas[1]
   - 0.0241035895650366 alphas[2] + 1.613389569355919 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 2126.80588765262 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 1542.344080484156
 R265: Q0_var[26] + Q_hat_adjusted_26 = 1889.486398120884
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 345.2099542181376
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.493774703007406 alphas[0] - 1.59429178579926 alphas[1]
   + 0.9975813314795949 alphas[2] + 0.1006978451805326 alphas[3]
   - alphas[4] + f_var[27] = 0
 R273: - 2126.80588765262 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 843.4913452079466
 R275: Q0_var[27] + Q_hat_adjusted_27 = 1038.180633144272
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 192.6742934923798
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: 0.0591950776359468 alphas[0] + 1.366840743853392 alphas[1]
   - 1.716674497466758 alphas[2] + 0.5824087203749593 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 2126.80588765262 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 2517.110863158761
 R285: Q0_var[28] + Q_hat_adjusted_28 = 3116.88800974837
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 597.9076464798175
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.4528779152734247 alphas[0] + 0.241420652828911 alphas[1]
   + 1.185101501114276 alphas[2] - 1.252326776685776 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 2126.80588765262 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 380.6188434464129
 R295: Q0_var[29] + Q_hat_adjusted_29 = 547.3175724150096
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 164.6778805194367
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 1.004200612610371 alphas[0] + 1.598652754974575 alphas[1]
   + 0.0394021874989427 alphas[2] + 1.781203516067779 alphas[3] - alphas[4]
   + f_var[30] = 0
 R303: - 2126.80588765262 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 1220.603621260045
 R305: Q0_var[30] + Q_hat_adjusted_30 = 1558.269680579169
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 335.7286596771327
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 700 Sold_0[30] - 700 Sold_1[30] + 700 Lost_0[30] + 700 Lost_1[30]
   + 300 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: - 0.8677863737108036 alphas[0] + 0.4666532660796253 alphas[1]
   + 1.390270369435714 alphas[2] - 1.47264783762299 alphas[3] - alphas[4]
   + f_var[31] = 0
 R313: - 2126.80588765262 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 663.2651171969653
 R315: Q0_var[31] + Q_hat_adjusted_31 = 799.4072648443339
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 134.046549620926
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 700 Sold_0[31] - 700 Sold_1[31] + 700 Lost_0[31] + 700 Lost_1[31]
   + 300 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: 0.3004959828634036 alphas[0] - 0.5277921768355862 alphas[1]
   - 1.542645421122971 alphas[2] - 1.075418018379352 alphas[3] - alphas[4]
   + f_var[32] = 0
 R323: - 2126.80588765262 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 2261.408770543923
 R325: Q0_var[32] + Q_hat_adjusted_32 = 2835.161187925172
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 571.9254286754098
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 700 Sold_0[32] - 700 Sold_1[32] + 700 Lost_0[32] + 700 Lost_1[32]
   + 300 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: - 0.1119083925633281 alphas[0] + 0.6926387673171881 alphas[1]
   - 1.613222600359389 alphas[2] + 1.897152438375739 alphas[3] - alphas[4]
   + f_var[33] = 0
 R333: - 2126.80588765262 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 2272.119003422546
 R335: Q0_var[33] + Q_hat_adjusted_33 = 2856.440231780068
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 582.4624704242198
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 700 Sold_0[33] - 700 Sold_1[33] + 700 Lost_0[33] + 700 Lost_1[33]
   + 300 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: 1.906869221658971 alphas[0] + 0.9391624402565898 alphas[1]
   - 0.7909220845559721 alphas[2] + 0.3503215840208569 alphas[3]
   - alphas[4] + f_var[34] = 0
 R343: - 2126.80588765262 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 1965.729566430107
 R345: Q0_var[34] + Q_hat_adjusted_34 = 2427.270712760948
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 459.6945310019868
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 700 Sold_0[34] - 700 Sold_1[34] + 700 Lost_0[34] + 700 Lost_1[34]
   + 300 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: - 0.2926198299344603 alphas[0] + 1.273631732713123 alphas[1]
   + 0.4461871409231427 alphas[2] - 0.4899266151287491 alphas[3]
   - alphas[4] + f_var[35] = 0
 R353: - 2126.80588765262 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 928.4540571325456
 R355: Q0_var[35] + Q_hat_adjusted_35 = 1205.404788218518
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 274.9964218150844
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 700 Sold_0[35] - 700 Sold_1[35] + 700 Lost_0[35] + 700 Lost_1[35]
   + 300 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 0.272272897315425 alphas[0] - 0.3674304596197978 alphas[1]
   + 0.766386846824618 alphas[2] - 0.5662856393435135 alphas[3] - alphas[4]
   + f_var[36] = 0
 R363: - 2126.80588765262 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 759.7102118519219
 R365: Q0_var[36] + Q_hat_adjusted_36 = 988.8645769031061
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 227.1912005588132
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 700 Sold_0[36] - 700 Sold_1[36] + 700 Lost_0[36] + 700 Lost_1[36]
   + 300 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.290043303091715 alphas[0] - 0.7875490923372758 alphas[1]
   + 0.7298126361507892 alphas[2] - 1.190530332343882 alphas[3] - alphas[4]
   + f_var[37] = 0
 R373: - 2126.80588765262 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 821.9589643620384
 R375: Q0_var[37] + Q_hat_adjusted_37 = 1056.536291160297
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 232.6516622685101
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 700 Sold_0[37] - 700 Sold_1[37] + 700 Lost_0[37] + 700 Lost_1[37]
   + 300 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: - 1.490392572539918 alphas[0] - 0.7046388338041588 alphas[1]
   + 1.192006260901824 alphas[2] - 0.795131890557693 alphas[3] - alphas[4]
   + f_var[38] = 0
 R383: - 2126.80588765262 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 1061.855717306719
 R385: Q0_var[38] + Q_hat_adjusted_38 = 1227.55893208776
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 163.647012710459
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 700 Sold_0[38] - 700 Sold_1[38] + 700 Lost_0[38] + 700 Lost_1[38]
   + 300 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 0.5283629773439216 alphas[0] - 1.689424358268716 alphas[1]
   - 0.1426216236649081 alphas[2] + 1.017927611535382 alphas[3] - alphas[4]
   + f_var[39] = 0
 R393: - 2126.80588765262 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 1478.681448633442
 R395: Q0_var[39] + Q_hat_adjusted_39 = 1843.504540836431
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 362.9044765383615
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 700 Sold_0[39] - 700 Sold_1[39] + 700 Lost_0[39] + 700 Lost_1[39]
   + 300 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: 0.6554521452111429 alphas[0] + 1.675506864806997 alphas[1]
   + 1.143367844985816 alphas[2] + 0.3226274065955959 alphas[3] - alphas[4]
   + f_var[40] = 0
 R403: - 2126.80588765262 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 667.4756315442539
 R405: Q0_var[40] + Q_hat_adjusted_40 = 840.4762996738787
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 170.9086378507388
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 700 Sold_0[40] - 700 Sold_1[40] + 700 Lost_0[40] + 700 Lost_1[40]
   + 300 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: 0.3707155139526951 alphas[0] - 0.6165802306829722 alphas[1]
   + 0.5575822881238133 alphas[2] - 0.4314347480666587 alphas[3]
   - alphas[4] + f_var[41] = 0
 R413: - 2126.80588765262 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 988.8189813450954
 R415: Q0_var[41] + Q_hat_adjusted_41 = 1249.181217067933
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 258.3653332767277
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 700 Sold_0[41] - 700 Sold_1[41] + 700 Lost_0[41] + 700 Lost_1[41]
   + 300 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.5864345217416338 alphas[0] - 0.4501242311450007 alphas[1]
   - 0.4805277801905115 alphas[2] + 1.65756481190855 alphas[3] - alphas[4]
   + f_var[42] = 0
 R423: - 2126.80588765262 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 2028.035016730847
 R425: Q0_var[42] + Q_hat_adjusted_42 = 2443.278088997075
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 413.3532379795524
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 700 Sold_0[42] - 700 Sold_1[42] + 700 Lost_0[42] + 700 Lost_1[42]
   + 300 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: 1.754680927335964 alphas[0] - 0.4086826359176909 alphas[1]
   - 0.8182316929044825 alphas[2] + 1.451560707369168 alphas[3] - alphas[4]
   + f_var[43] = 0
 R433: - 2126.80588765262 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 1797.384902998831
 R435: Q0_var[43] + Q_hat_adjusted_43 = 2263.052651616658
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 463.7718047953749
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 700 Sold_0[43] - 700 Sold_1[43] + 700 Lost_0[43] + 700 Lost_1[43]
   + 300 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: - 0.4730729541736584 alphas[0] - 1.73889161897447 alphas[1]
   + 1.537025024064408 alphas[2] + 0.3871213365932374 alphas[3] - alphas[4]
   + f_var[44] = 0
 R443: - 2126.80588765262 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 559.0831405079221
 R445: Q0_var[44] + Q_hat_adjusted_44 = 673.2008097432926
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 112.136348357214
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 700 Sold_0[44] - 700 Sold_1[44] + 700 Lost_0[44] + 700 Lost_1[44]
   + 300 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: - 0.4872912432374461 alphas[0] + 1.625117921935653 alphas[1]
   + 0.1778544627265278 alphas[2] + 1.064964331491352 alphas[3] - alphas[4]
   + f_var[45] = 0
 R453: - 2126.80588765262 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 1120.259712044138
 R455: Q0_var[45] + Q_hat_adjusted_45 = 1437.262728372915
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 315.05799218837
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 700 Sold_0[45] - 700 Sold_1[45] + 700 Lost_0[45] + 700 Lost_1[45]
   + 300 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: 1.20317178882866 alphas[0] - 0.7918609798731873 alphas[1]
   - 0.5656531491045651 alphas[2] - 1.027611878120909 alphas[3] - alphas[4]
   + f_var[46] = 0
 R463: - 2126.80588765262 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 1849.465263068239
 R465: Q0_var[46] + Q_hat_adjusted_46 = 2277.41784394868
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 426.0622974210695
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 700 Sold_0[46] - 700 Sold_1[46] + 700 Lost_0[46] + 700 Lost_1[46]
   + 300 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: 1.502350960657328 alphas[0] - 1.208659545287354 alphas[1]
   - 1.239209357565287 alphas[2] - 0.00310858278931209 alphas[3]
   - alphas[4] + f_var[47] = 0
 R473: - 2126.80588765262 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 2094.362770942495
 R475: Q0_var[47] + Q_hat_adjusted_47 = 2622.774084166058
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 526.6229880792564
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 700 Sold_0[47] - 700 Sold_1[47] + 700 Lost_0[47] + 700 Lost_1[47]
   + 300 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: 0.8173510907951662 alphas[0] - 0.1279660997613337 alphas[1]
   + 0.0264500408806964 alphas[2] + 0.8780000925996656 alphas[3]
   - alphas[4] + f_var[48] = 0
 R483: - 2126.80588765262 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 1437.196003413942
 R485: Q0_var[48] + Q_hat_adjusted_48 = 1776.794872195123
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 337.6623911060412
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 700 Sold_0[48] - 700 Sold_1[48] + 700 Lost_0[48] + 700 Lost_1[48]
   + 300 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: 0.6400155019933705 alphas[0] + 0.4974024511025514 alphas[1]
   - 1.627559452791979 alphas[2] - 0.9767125594686354 alphas[3] - alphas[4]
   + f_var[49] = 0
 R493: - 2126.80588765262 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 2134.024042868702
 R495: Q0_var[49] + Q_hat_adjusted_49 = 2720.401706310468
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 584.6029358588427
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 700 Sold_0[49] - 700 Sold_1[49] + 700 Lost_0[49] + 700 Lost_1[49]
   + 300 Left_1[49] + profits_vars[49] = 0
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
 Q0_var[0] <= 2127.80588765262
 Q0_var[1] <= 2127.80588765262
 Q0_var[2] <= 2127.80588765262
 Q0_var[3] <= 2127.80588765262
 Q0_var[4] <= 2127.80588765262
 Q0_var[5] <= 2127.80588765262
 Q0_var[6] <= 2127.80588765262
 Q0_var[7] <= 2127.80588765262
 Q0_var[8] <= 2127.80588765262
 Q0_var[9] <= 2127.80588765262
 Q0_var[10] <= 2127.80588765262
 Q0_var[11] <= 2127.80588765262
 Q0_var[12] <= 2127.80588765262
 Q0_var[13] <= 2127.80588765262
 Q0_var[14] <= 2127.80588765262
 Q0_var[15] <= 2127.80588765262
 Q0_var[16] <= 2127.80588765262
 Q0_var[17] <= 2127.80588765262
 Q0_var[18] <= 2127.80588765262
 Q0_var[19] <= 2127.80588765262
 Q0_var[20] <= 2127.80588765262
 Q0_var[21] <= 2127.80588765262
 Q0_var[22] <= 2127.80588765262
 Q0_var[23] <= 2127.80588765262
 Q0_var[24] <= 2127.80588765262
 Q0_var[25] <= 2127.80588765262
 Q0_var[26] <= 2127.80588765262
 Q0_var[27] <= 2127.80588765262
 Q0_var[28] <= 2127.80588765262
 Q0_var[29] <= 2127.80588765262
 Q0_var[30] <= 2127.80588765262
 Q0_var[31] <= 2127.80588765262
 Q0_var[32] <= 2127.80588765262
 Q0_var[33] <= 2127.80588765262
 Q0_var[34] <= 2127.80588765262
 Q0_var[35] <= 2127.80588765262
 Q0_var[36] <= 2127.80588765262
 Q0_var[37] <= 2127.80588765262
 Q0_var[38] <= 2127.80588765262
 Q0_var[39] <= 2127.80588765262
 Q0_var[40] <= 2127.80588765262
 Q0_var[41] <= 2127.80588765262
 Q0_var[42] <= 2127.80588765262
 Q0_var[43] <= 2127.80588765262
 Q0_var[44] <= 2127.80588765262
 Q0_var[45] <= 2127.80588765262
 Q0_var[46] <= 2127.80588765262
 Q0_var[47] <= 2127.80588765262
 Q0_var[48] <= 2127.80588765262
 Q0_var[49] <= 2127.80588765262
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
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1477.404818848541 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 396.4401353905788 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 325.4521829879686 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 105.0318478275103 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1171.127177238832 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 337.9130979286114 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 1891.966029414279 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 454.3851954705243 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 978.3055855823229 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 121.9877160107273 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 2174.401044502069 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 539.760741511865 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 1528.607093475993 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 360.0407083178366 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 322.1606186615509 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 115.3305241628722 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 711.607186804144 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 212.2068059604201 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 2228.36595003303 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 576.8378482136565 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 1960.353364754076 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 391.1598665260277 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 894.1038050098194 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 153.7362838843955 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 1052.059162567585 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 243.7722511402829 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1413.402374621745 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 328.3518129302421 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 487.9585269049063 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 110.4750346366098 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 1236.303307498596 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 305.8077568068794 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1174.194375792902 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 344.7293177173491 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 751.4523840897484 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 221.8389376406162 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1664.311079507421 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 394.3195001459933 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1824.739408660224 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 476.6200597960884 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 698.8991175480789 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 217.917112052819 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1492.765652653201 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 410.2499501399939 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 1823.211351936723 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 419.8111214818708 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 2446.584764739198 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 574.2701505644205 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1989.077137065633 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 489.1380836482219 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 2080.988378515591 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 524.1726348950031 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 1542.344080484156 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 345.2099542181376 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 843.4913452079466 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 192.6742934923798 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 2517.110863158761 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 597.9076464798175 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 380.6188434464129 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 164.6778805194367 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 1220.603621260045 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 335.7286596771327 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 663.2651171969653 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 134.046549620926 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 2261.408770543923 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 571.9254286754098 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 2272.119003422546 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 582.4624704242198 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 1965.729566430107 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 459.6945310019868 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 928.4540571325456 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 274.9964218150844 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 759.7102118519219 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 227.1912005588132 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 821.9589643620384 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 232.6516622685101 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 1061.855717306719 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 163.647012710459 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 1478.681448633442 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 362.9044765383615 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 667.4756315442539 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 170.9086378507388 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 988.8189813450954 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 258.3653332767277 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 2028.035016730847 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 413.3532379795524 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 1797.384902998831 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 463.7718047953749 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 559.0831405079221 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 112.136348357214 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 1120.259712044138 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 315.05799218837 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 1849.465263068239 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 426.0622974210695 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 2094.362770942495 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 526.6229880792564 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 1437.196003413942 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 337.6623911060412 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 2134.024042868702 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 584.6029358588427 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
End
