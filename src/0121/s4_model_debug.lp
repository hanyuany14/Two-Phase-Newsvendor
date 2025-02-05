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
 R0: - 0.5707452599980976 alphas[0] + 1.070471684459597 alphas[1]
   + 1.500938953723704 alphas[2] + 0.2626276896697152 alphas[3] - alphas[4]
   + f_var[0] = 0
 Q0_upper_bound_0: - 1313.714597627199 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 R2: Q0_var[0] + Q_hat_adjusted_0 = 583.1395417484991
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: Q0_var[0] + Lost_0_aux_0 = 455.7734011299941
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: Q1_plus_left[0] + Lost_0_aux_0
   = 127.3800543371205
 Profit_Constraint_0: - 800 Sold_0[0] + 800 Lost_0[0] - 800 Sold_1[0]
   + 200 Left_1[0] + 800 Lost_1[0] + profits_vars[0] = 0
 R9: 0.9191423420362758 alphas[0] + 1.56533293035073 alphas[1]
   + 0.9479654598139138 alphas[2] - 1.638420838139021 alphas[3] - alphas[4]
   + f_var[1] = 0
 Q0_upper_bound_1: - 1313.714597627199 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 R11: Q0_var[1] + Q_hat_adjusted_1 = 774.5674617949239
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: Q0_var[1] + Lost_0_aux_1 = 433.4550214245154
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: Q1_plus_left[1] + Lost_0_aux_1
   = 350.703854210086
 Profit_Constraint_1: - 800 Sold_0[1] + 800 Lost_0[1] - 800 Sold_1[1]
   + 200 Left_1[1] + 800 Lost_1[1] + profits_vars[1] = 0
 R18: - 0.153009161028938 alphas[0] - 0.5757249402826893 alphas[1]
   + 0.8779801990896655 alphas[2] - 1.460411039911365 alphas[3] - alphas[4]
   + f_var[2] = 0
 Q0_upper_bound_2: - 1313.714597627199 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 R20: Q0_var[2] + Q_hat_adjusted_2 = 772.5405172156788
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: Q0_var[2] + Lost_0_aux_2 = 610.1860591594664
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: Q1_plus_left[2] + Lost_0_aux_2
   = 162.3591764004619
 Profit_Constraint_2: - 800 Sold_0[2] + 800 Lost_0[2] - 800 Sold_1[2]
   + 200 Left_1[2] + 800 Lost_1[2] + profits_vars[2] = 0
 R27: 1.713903709337297 alphas[0] - 0.1963332588194304 alphas[1]
   + 1.249577144329202 alphas[2] - 1.003836333639178 alphas[3] - alphas[4]
   + f_var[3] = 0
 Q0_upper_bound_3: - 1313.714597627199 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 R29: Q0_var[3] + Q_hat_adjusted_3 = 720.750065167382
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: Q0_var[3] + Lost_0_aux_3 = 169.823119790778
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: Q1_plus_left[3] + Lost_0_aux_3
   = 575.7356520491599
 Profit_Constraint_3: - 800 Sold_0[3] + 800 Lost_0[3] - 800 Sold_1[3]
   + 200 Left_1[3] + 800 Lost_1[3] + profits_vars[3] = 0
 R36: - 0.058029171781454 alphas[0] - 0.5136618317864551 alphas[1]
   - 1.385866378777106 alphas[2] + 0.7643859794712374 alphas[3] - alphas[4]
   + f_var[4] = 0
 Q0_upper_bound_4: - 1313.714597627199 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 R38: Q0_var[4] + Q_hat_adjusted_4 = 1485.019901491553
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: Q0_var[4] + Lost_0_aux_4 = 1195.540081262036
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: Q1_plus_left[4] + Lost_0_aux_4
   = 289.4741180977849
 Profit_Constraint_4: - 800 Sold_0[4] + 800 Lost_0[4] - 800 Sold_1[4]
   + 200 Left_1[4] + 800 Lost_1[4] + profits_vars[4] = 0
 R45: - 0.228327859341396 alphas[0] + 1.282925868061986 alphas[1]
   + 0.7219094211679499 alphas[2] - 0.5284607052963116 alphas[3]
   - alphas[4] + f_var[5] = 0
 Q0_upper_bound_5: - 1313.714597627199 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 R47: Q0_var[5] + Q_hat_adjusted_5 = 796.5055875855871
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: Q0_var[5] + Lost_0_aux_5 = 625.3860526814287
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: Q1_plus_left[5] + Lost_0_aux_5
   = 171.1225475699914
 Profit_Constraint_5: - 800 Sold_0[5] + 800 Lost_0[5] - 800 Sold_1[5]
   + 200 Left_1[5] + 800 Lost_1[5] + profits_vars[5] = 0
 R54: - 0.0828495419217684 alphas[0] - 1.168784647874499 alphas[1]
   + 1.215999540787469 alphas[2] + 1.252070652327586 alphas[3] - alphas[4]
   + f_var[6] = 0
 Q0_upper_bound_6: - 1313.714597627199 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 R56: Q0_var[6] + Q_hat_adjusted_6 = 798.1717775834824
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: Q0_var[6] + Lost_0_aux_6 = 575.5844350405919
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: Q1_plus_left[6] + Lost_0_aux_6
   = 205.4102025365846
 Profit_Constraint_6: - 800 Sold_0[6] + 800 Lost_0[6] - 800 Sold_1[6]
   + 200 Left_1[6] + 800 Lost_1[6] + profits_vars[6] = 0
 R63: - 0.3954222152509145 alphas[0] - 0.2698811955727397 alphas[1]
   + 0.4758800191394995 alphas[2] - 0.99923761631174 alphas[3] - alphas[4]
   + f_var[7] = 0
 Q0_upper_bound_7: - 1313.714597627199 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 R65: Q0_var[7] + Q_hat_adjusted_7 = 865.614894722304
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: Q0_var[7] + Lost_0_aux_7 = 680.6800123025196
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: Q1_plus_left[7] + Lost_0_aux_7
   = 184.9370937503554
 Profit_Constraint_7: - 800 Sold_0[7] + 800 Lost_0[7] - 800 Sold_1[7]
   + 200 Left_1[7] + 800 Lost_1[7] + profits_vars[7] = 0
 R72: - 0.9691858115137737 alphas[0] - 0.456733400221593 alphas[1]
   - 0.8539917292647438 alphas[2] + 0.6998249972916352 alphas[3]
   - alphas[4] + f_var[8] = 0
 Q0_upper_bound_8: - 1313.714597627199 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 R74: Q0_var[8] + Q_hat_adjusted_8 = 1342.901417017237
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: Q0_var[8] + Lost_0_aux_8 = 1083.2898673318
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: Q1_plus_left[8] + Lost_0_aux_8
   = 259.6093667825127
 Profit_Constraint_8: - 800 Sold_0[8] + 800 Lost_0[8] - 800 Sold_1[8]
   + 200 Left_1[8] + 800 Lost_1[8] + profits_vars[8] = 0
 R81: - 1.701281133939507 alphas[0] + 1.237038529169367 alphas[1]
   - 0.7374410922516981 alphas[2] + 0.2398804333632175 alphas[3]
   - alphas[4] + f_var[9] = 0
 Q0_upper_bound_9: - 1313.714597627199 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 R83: Q0_var[9] + Q_hat_adjusted_9 = 1360.162750686656
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: Q0_var[9] + Lost_0_aux_9 = 1107.09836748275
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: Q1_plus_left[9] + Lost_0_aux_9
   = 253.0650507613931
 Profit_Constraint_9: - 800 Sold_0[9] + 800 Lost_0[9] - 800 Sold_1[9]
   + 200 Left_1[9] + 800 Lost_1[9] + profits_vars[9] = 0
 R90: - 1.008621214769357 alphas[0] + 0.2357816597478171 alphas[1]
   - 1.538914897609143 alphas[2] - 0.9748597985386164 alphas[3] - alphas[4]
   + f_var[10] = 0
 Q0_upper_bound_10:
   - 1313.714597627199 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 R92: Q0_var[10] + Q_hat_adjusted_10 = 1609.052572917187
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: Q0_var[10] + Lost_0_aux_10 = 1310.987616267015
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: Q1_plus_left[10] + Lost_0_aux_10
   = 298.0677894793409
 Profit_Constraint_10: - 800 Sold_0[10] + 800 Lost_0[10] - 800 Sold_1[10]
   + 200 Left_1[10] + 800 Lost_1[10] + profits_vars[10] = 0
 R99: - 0.9470110948883567 alphas[0] + 0.3003552879829094 alphas[1]
   + 1.228768937980696 alphas[2] - 1.039010042019775 alphas[3] - alphas[4]
   + f_var[11] = 0
 Q0_upper_bound_11:
   - 1313.714597627199 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 R101: Q0_var[11] + Q_hat_adjusted_11 = 751.431885570822
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: Q0_var[11] + Lost_0_aux_11 = 608.7737460707782
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: Q1_plus_left[11] + Lost_0_aux_11
   = 142.6623954734695
 Profit_Constraint_11: - 800 Sold_0[11] + 800 Lost_0[11] - 800 Sold_1[11]
   + 200 Left_1[11] + 800 Lost_1[11] + profits_vars[11] = 0
 R108: - 1.111713139789012 alphas[0] + 1.767436110441344 alphas[1]
   + 0.9850711574655432 alphas[2] - 0.8091235572460596 alphas[3]
   - alphas[4] + f_var[12] = 0
 Q0_upper_bound_12:
   - 1313.714597627199 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 R110: Q0_var[12] + Q_hat_adjusted_12 = 777.6218917360526
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: Q0_var[12] + Lost_0_aux_12 = 621.2755875215321
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: Q1_plus_left[12] + Lost_0_aux_12
   = 156.3460211507899
 Profit_Constraint_12: - 800 Sold_0[12] + 800 Lost_0[12] - 800 Sold_1[12]
   + 200 Left_1[12] + 800 Lost_1[12] + profits_vars[12] = 0
 R117: 0.8879134194927919 alphas[0] - 0.1932933445139454 alphas[1]
   - 0.0932763658028453 alphas[2] + 1.316654102689879 alphas[3] - alphas[4]
   + f_var[13] = 0
 Q0_upper_bound_13:
   - 1313.714597627199 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 R119: Q0_var[13] + Q_hat_adjusted_13 = 1119.234781498868
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: Q0_var[13] + Lost_0_aux_13 = 902.339313510209
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: Q1_plus_left[13] + Lost_0_aux_13
   = 216.8952137911535
 Profit_Constraint_13: - 800 Sold_0[13] + 800 Lost_0[13] - 800 Sold_1[13]
   + 200 Left_1[13] + 800 Lost_1[13] + profits_vars[13] = 0
 R126: 1.361890421837527 alphas[0] + 1.696642565720933 alphas[1]
   - 1.125554183019969 alphas[2] - 0.3335396751466935 alphas[3] - alphas[4]
   + f_var[14] = 0
 Q0_upper_bound_14:
   - 1313.714597627199 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 R128: Q0_var[14] + Q_hat_adjusted_14 = 1442.891820603423
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: Q0_var[14] + Lost_0_aux_14 = 1168.030156567468
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: Q1_plus_left[14] + Lost_0_aux_14
   = 274.8575933249894
 Profit_Constraint_14: - 800 Sold_0[14] + 800 Lost_0[14] - 800 Sold_1[14]
   + 200 Left_1[14] + 800 Lost_1[14] + profits_vars[14] = 0
 R135: 0.8453007902672948 alphas[0] - 0.3511165583720055 alphas[1]
   + 1.460901222210551 alphas[2] - 1.468582025062098 alphas[3] - alphas[4]
   + f_var[15] = 0
 Q0_upper_bound_15:
   - 1313.714597627199 Fraction_for_second_order_amount[15] + Q0_var[15]
   = 0
 R137: Q0_var[15] + Q_hat_adjusted_15 = 700.4720774189623
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: Q0_var[15] + Lost_0_aux_15 = 446.9623868488507
 Constr_Q1_plus_left_15: - Left_0[15] + Q1_plus_left[15] - Q1_var[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: Q1_plus_left[15] + Lost_0_aux_15
   = 227.7165654975522
 Profit_Constraint_15: - 800 Sold_0[15] + 800 Lost_0[15] - 800 Sold_1[15]
   + 200 Left_1[15] + 800 Lost_1[15] + profits_vars[15] = 0
 R144: 0.1733801433293181 alphas[0] + 1.332083314898666 alphas[1]
   + 0.1427890937964064 alphas[2] + 1.573802972311753 alphas[3] - alphas[4]
   + f_var[16] = 0
 Q0_upper_bound_16:
   - 1313.714597627199 Fraction_for_second_order_amount[16] + Q0_var[16]
   = 0
 R146: Q0_var[16] + Q_hat_adjusted_16 = 1010.251119288836
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: Q0_var[16] + Lost_0_aux_16 = 806.6113964906422
 Constr_Q1_plus_left_16: - Left_0[16] + Q1_plus_left[16] - Q1_var[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: Q1_plus_left[16] + Lost_0_aux_16
   = 203.6401427861327
 Profit_Constraint_16: - 800 Sold_0[16] + 800 Lost_0[16] - 800 Sold_1[16]
   + 200 Left_1[16] + 800 Lost_1[16] + profits_vars[16] = 0
 R153: - 0.215664006642524 alphas[0] - 1.477659082794586 alphas[1]
   + 0.7517552726192818 alphas[2] - 1.154069947424716 alphas[3] - alphas[4]
   + f_var[17] = 0
 Q0_upper_bound_17:
   - 1313.714597627199 Fraction_for_second_order_amount[17] + Q0_var[17]
   = 0
 R155: Q0_var[17] + Q_hat_adjusted_17 = 864.0329240888593
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: Q0_var[17] + Lost_0_aux_17 = 586.5977877510986
 Constr_Q1_plus_left_17: - Left_0[17] + Q1_plus_left[17] - Q1_var[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: Q1_plus_left[17] + Lost_0_aux_17
   = 236.436415247552
 Profit_Constraint_17: - 800 Sold_0[17] + 800 Lost_0[17] - 800 Sold_1[17]
   + 200 Left_1[17] + 800 Lost_1[17] + profits_vars[17] = 0
 R162: 1.529875692474957 alphas[0] - 1.05406884141132 alphas[1]
   + 1.915005960445386 alphas[2] - 0.6230092407095527 alphas[3] - alphas[4]
   + f_var[18] = 0
 Q0_upper_bound_18:
   - 1313.714597627199 Fraction_for_second_order_amount[18] + Q0_var[18]
   = 0
 R164: Q0_var[18] + Q_hat_adjusted_18 = 490.3376896960793
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: Q0_var[18] + Lost_0_aux_18 = 386.2125767106902
 Constr_Q1_plus_left_18: - Left_0[18] + Q1_plus_left[18] - Q1_var[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: Q1_plus_left[18] + Lost_0_aux_18
   = 104.1301995882807
 Profit_Constraint_18: - 800 Sold_0[18] + 800 Lost_0[18] - 800 Sold_1[18]
   + 200 Left_1[18] + 800 Lost_1[18] + profits_vars[18] = 0
 R171: - 1.493190803936965 alphas[0] + 0.7113082207884873 alphas[1]
   - 0.2891676082870125 alphas[2] + 0.3540353763417799 alphas[3]
   - alphas[4] + f_var[19] = 0
 Q0_upper_bound_19:
   - 1313.714597627199 Fraction_for_second_order_amount[19] + Q0_var[19]
   = 0
 R173: Q0_var[19] + Q_hat_adjusted_19 = 1084.911230868678
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: Q0_var[19] + Lost_0_aux_19 = 857.0158397214361
 Constr_Q1_plus_left_19: - Left_0[19] + Q1_plus_left[19] - Q1_var[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: Q1_plus_left[19] + Lost_0_aux_19
   = 227.8945032797447
 Profit_Constraint_19: - 800 Sold_0[19] + 800 Lost_0[19] - 800 Sold_1[19]
   + 200 Left_1[19] + 800 Lost_1[19] + profits_vars[19] = 0
 R180: - 1.284373436012665 alphas[0] - 0.6274883583176433 alphas[1]
   - 0.9715398420184412 alphas[2] - 0.8193956803154996 alphas[3]
   - alphas[4] + f_var[20] = 0
 Q0_upper_bound_20:
   - 1313.714597627199 Fraction_for_second_order_amount[20] + Q0_var[20]
   = 0
 R182: Q0_var[20] + Q_hat_adjusted_20 = 1306.527318078063
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: Q0_var[20] + Lost_0_aux_20 = 1040.317914110936
 Constr_Q1_plus_left_20: - Left_0[20] + Q1_plus_left[20] - Q1_var[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: Q1_plus_left[20] + Lost_0_aux_20
   = 266.209691047238
 Profit_Constraint_20: - 800 Sold_0[20] + 800 Lost_0[20] - 800 Sold_1[20]
   + 200 Left_1[20] + 800 Lost_1[20] + profits_vars[20] = 0
 R189: 0.3195155197425125 alphas[0] + 0.7882315413264702 alphas[1]
   - 0.6146647472016619 alphas[2] + 0.420122968063511 alphas[3] - alphas[4]
   + f_var[21] = 0
 Q0_upper_bound_21:
   - 1313.714597627199 Fraction_for_second_order_amount[21] + Q0_var[21]
   = 0
 R191: Q0_var[21] + Q_hat_adjusted_21 = 1097.050958997455
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: Q0_var[21] + Lost_0_aux_21 = 850.8794252886834
 Constr_Q1_plus_left_21: - Left_0[21] + Q1_plus_left[21] - Q1_var[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: Q1_plus_left[21] + Lost_0_aux_21
   = 246.1711613540316
 Profit_Constraint_21: - 800 Sold_0[21] + 800 Lost_0[21] - 800 Sold_1[21]
   + 200 Left_1[21] + 800 Lost_1[21] + profits_vars[21] = 0
 R198: 0.4989818202668497 alphas[0] - 1.705019825650084 alphas[1]
   + 0.2501602198252286 alphas[2] - 1.6492321372128 alphas[3] - alphas[4]
   + f_var[22] = 0
 Q0_upper_bound_22:
   - 1313.714597627199 Fraction_for_second_order_amount[22] + Q0_var[22]
   = 0
 R200: Q0_var[22] + Q_hat_adjusted_22 = 932.5238650313506
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: Q0_var[22] + Lost_0_aux_22 = 648.0700582895297
 Constr_Q1_plus_left_22: - Left_0[22] + Q1_plus_left[22] - Q1_var[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: Q1_plus_left[22] + Lost_0_aux_22
   = 278.6687724412353
 Profit_Constraint_22: - 800 Sold_0[22] + 800 Lost_0[22] - 800 Sold_1[22]
   + 200 Left_1[22] + 800 Lost_1[22] + profits_vars[22] = 0
 R207: 1.709262187672383 alphas[0] + 0.9138884121141941 alphas[1]
   - 1.082467973378304 alphas[2] + 0.5030393298699345 alphas[3] - alphas[4]
   + f_var[23] = 0
 Q0_upper_bound_23:
   - 1313.714597627199 Fraction_for_second_order_amount[23] + Q0_var[23]
   = 0
 R209: Q0_var[23] + Q_hat_adjusted_23 = 1334.811536066645
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: Q0_var[23] + Lost_0_aux_23 = 1062.368340656185
 Constr_Q1_plus_left_23: - Left_0[23] + Q1_plus_left[23] - Q1_var[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: Q1_plus_left[23] + Lost_0_aux_23
   = 272.4383034843366
 Profit_Constraint_23: - 800 Sold_0[23] + 800 Lost_0[23] - 800 Sold_1[23]
   + 200 Left_1[23] + 800 Lost_1[23] + profits_vars[23] = 0
 R216: - 1.139704306831707 alphas[0] + 1.60451270122029 alphas[1]
   - 1.476868720031216 alphas[2] + 1.718494021422222 alphas[3] - alphas[4]
   + f_var[24] = 0
 Q0_upper_bound_24:
   - 1313.714597627199 Fraction_for_second_order_amount[24] + Q0_var[24]
   = 0
 R218: Q0_var[24] + Q_hat_adjusted_24 = 1515.384798833413
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: Q0_var[24] + Lost_0_aux_24 = 1220.801283411015
 Constr_Q1_plus_left_24: - Left_0[24] + Q1_plus_left[24] - Q1_var[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: Q1_plus_left[24] + Lost_0_aux_24
   = 294.5838978046261
 Profit_Constraint_24: - 800 Sold_0[24] + 800 Lost_0[24] - 800 Sold_1[24]
   + 200 Left_1[24] + 800 Lost_1[24] + profits_vars[24] = 0
 R225: 1.441598087339471 alphas[0] + 1.204898038961154 alphas[1]
   + 0.2588109252509967 alphas[2] - 0.6872039383535031 alphas[3]
   - alphas[4] + f_var[25] = 0
 Q0_upper_bound_25:
   - 1313.714597627199 Fraction_for_second_order_amount[25] + Q0_var[25]
   = 0
 R227: Q0_var[25] + Q_hat_adjusted_25 = 1036.980677090709
 Constr_Left_0_diff_aux_25: Sold_0[25] - Q0_var[25] + Left_0_aux_25 = 0
 Constr_Lost_0_diff_aux_25: Q0_var[25] + Lost_0_aux_25 = 839.8568396032224
 Constr_Q1_plus_left_25: - Left_0[25] + Q1_plus_left[25] - Q1_var[25] = 0
 Constr_Left_1_diff_aux_25: Sold_1[25] - Q1_plus_left[25] + Left_0_aux_25
   = 0
 Constr_Lost_1_diff_aux_25: Q1_plus_left[25] + Lost_0_aux_25
   = 197.1255191869246
 Profit_Constraint_25: - 800 Sold_0[25] + 800 Lost_0[25] - 800 Sold_1[25]
   + 200 Left_1[25] + 800 Lost_1[25] + profits_vars[25] = 0
 R234: - 1.666919159946136 alphas[0] - 0.7425975757923894 alphas[1]
   - 0.2518009959635112 alphas[2] + 0.8114894152955283 alphas[3]
   - alphas[4] + f_var[26] = 0
 Q0_upper_bound_26:
   - 1313.714597627199 Fraction_for_second_order_amount[26] + Q0_var[26]
   = 0
 R236: Q0_var[26] + Q_hat_adjusted_26 = 1102.301456752519
 Constr_Left_0_diff_aux_26: Sold_0[26] - Q0_var[26] + Left_0_aux_26 = 0
 Constr_Lost_0_diff_aux_26: Q0_var[26] + Lost_0_aux_26 = 750.3761583847518
 Constr_Q1_plus_left_26: - Left_0[26] + Q1_plus_left[26] - Q1_var[26] = 0
 Constr_Left_1_diff_aux_26: Sold_1[26] - Q1_plus_left[26] + Left_0_aux_26
   = 0
 Constr_Lost_1_diff_aux_26: Q1_plus_left[26] + Lost_0_aux_26
   = 323.8468923707702
 Profit_Constraint_26: - 800 Sold_0[26] + 800 Lost_0[26] - 800 Sold_1[26]
   + 200 Left_1[26] + 800 Lost_1[26] + profits_vars[26] = 0
 R243: 0.7607066426934646 alphas[0] - 0.7928886946974023 alphas[1]
   + 0.1673010572771918 alphas[2] + 1.65787410506406 alphas[3] - alphas[4]
   + f_var[27] = 0
 Q0_upper_bound_27:
   - 1313.714597627199 Fraction_for_second_order_amount[27] + Q0_var[27]
   = 0
 R245: Q0_var[27] + Q_hat_adjusted_27 = 998.2628727683925
 Constr_Left_0_diff_aux_27: Sold_0[27] - Q0_var[27] + Left_0_aux_27 = 0
 Constr_Lost_0_diff_aux_27: Q0_var[27] + Lost_0_aux_27 = 795.999852778667
 Constr_Q1_plus_left_27: - Left_0[27] + Q1_plus_left[27] - Q1_var[27] = 0
 Constr_Left_1_diff_aux_27: Sold_1[27] - Q1_plus_left[27] + Left_0_aux_27
   = 0
 Constr_Lost_1_diff_aux_27: Q1_plus_left[27] + Lost_0_aux_27
   = 202.2637965092961
 Profit_Constraint_27: - 800 Sold_0[27] + 800 Lost_0[27] - 800 Sold_1[27]
   + 200 Left_1[27] + 800 Lost_1[27] + profits_vars[27] = 0
 R252: - 1.66037731766152 alphas[0] + 1.10511144870091 alphas[1]
   + 0.7840778699883453 alphas[2] - 0.9255739601548184 alphas[3]
   - alphas[4] + f_var[28] = 0
 Q0_upper_bound_28:
   - 1313.714597627199 Fraction_for_second_order_amount[28] + Q0_var[28]
   = 0
 R254: Q0_var[28] + Q_hat_adjusted_28 = 840.5708048395524
 Constr_Left_0_diff_aux_28: Sold_0[28] - Q0_var[28] + Left_0_aux_28 = 0
 Constr_Lost_0_diff_aux_28: Q0_var[28] + Lost_0_aux_28 = 672.9453731245621
 Constr_Q1_plus_left_28: - Left_0[28] + Q1_plus_left[28] - Q1_var[28] = 0
 Constr_Left_1_diff_aux_28: Sold_1[28] - Q1_plus_left[28] + Left_0_aux_28
   = 0
 Constr_Lost_1_diff_aux_28: Q1_plus_left[28] + Lost_0_aux_28
   = 167.631790367369
 Profit_Constraint_28: - 800 Sold_0[28] + 800 Lost_0[28] - 800 Sold_1[28]
   + 200 Left_1[28] + 800 Lost_1[28] + profits_vars[28] = 0
 R261: 1.356716689984285 alphas[0] - 1.456085953663965 alphas[1]
   - 0.8581083947560578 alphas[2] - 1.64827964516296 alphas[3] - alphas[4]
   + f_var[29] = 0
 Q0_upper_bound_29:
   - 1313.714597627199 Fraction_for_second_order_amount[29] + Q0_var[29]
   = 0
 R263: Q0_var[29] + Q_hat_adjusted_29 = 1360.220492081367
 Constr_Left_0_diff_aux_29: Sold_0[29] - Q0_var[29] + Left_0_aux_29 = 0
 Constr_Lost_0_diff_aux_29: Q0_var[29] + Lost_0_aux_29 = 824.1541446040673
 Constr_Q1_plus_left_29: - Left_0[29] + Q1_plus_left[29] - Q1_var[29] = 0
 Constr_Left_1_diff_aux_29: Sold_1[29] - Q1_plus_left[29] + Left_0_aux_29
   = 0
 Constr_Lost_1_diff_aux_29: Q1_plus_left[29] + Lost_0_aux_29
   = 528.2279131654524
 Profit_Constraint_29: - 800 Sold_0[29] + 800 Lost_0[29] - 800 Sold_1[29]
   + 200 Left_1[29] + 800 Lost_1[29] + profits_vars[29] = 0
 R270: - 0.4739189537443609 alphas[0] - 0.3244777828317514 alphas[1]
   - 1.405175615979193 alphas[2] + 0.1479723431375203 alphas[3] - alphas[4]
   + f_var[30] = 0
 Q0_upper_bound_30:
   - 1313.714597627199 Fraction_for_second_order_amount[30] + Q0_var[30]
   = 0
 R272: Q0_var[30] + Q_hat_adjusted_30 = 1448.082451872824
 Constr_Left_0_diff_aux_30: Sold_0[30] - Q0_var[30] + Left_0_aux_30 = 0
 Constr_Lost_0_diff_aux_30: Q0_var[30] + Lost_0_aux_30 = 1157.522389180815
 Constr_Q1_plus_left_30: - Left_0[30] + Q1_plus_left[30] - Q1_var[30] = 0
 Constr_Left_1_diff_aux_30: Sold_1[30] - Q1_plus_left[30] + Left_0_aux_30
   = 0
 Constr_Lost_1_diff_aux_30: Q1_plus_left[30] + Lost_0_aux_30
   = 290.5583313943889
 Profit_Constraint_30: - 800 Sold_0[30] + 800 Lost_0[30] - 800 Sold_1[30]
   + 200 Left_1[30] + 800 Lost_1[30] + profits_vars[30] = 0
 R279: - 0.3929513073548907 alphas[0] - 0.4891337726008566 alphas[1]
   + 0.3373583131120997 alphas[2] + 0.441954755862285 alphas[3] - alphas[4]
   + f_var[31] = 0
 Q0_upper_bound_31:
   - 1313.714597627199 Fraction_for_second_order_amount[31] + Q0_var[31]
   = 0
 R281: Q0_var[31] + Q_hat_adjusted_31 = 930.9756695868571
 Constr_Left_0_diff_aux_31: Sold_0[31] - Q0_var[31] + Left_0_aux_31 = 0
 Constr_Lost_0_diff_aux_31: Q0_var[31] + Lost_0_aux_31 = 738.2604881943967
 Constr_Q1_plus_left_31: - Left_0[31] + Q1_plus_left[31] - Q1_var[31] = 0
 Constr_Left_1_diff_aux_31: Sold_1[31] - Q1_plus_left[31] + Left_0_aux_31
   = 0
 Constr_Lost_1_diff_aux_31: Q1_plus_left[31] + Lost_0_aux_31
   = 192.7150848729526
 Profit_Constraint_31: - 800 Sold_0[31] + 800 Lost_0[31] - 800 Sold_1[31]
   + 200 Left_1[31] + 800 Lost_1[31] + profits_vars[31] = 0
 R288: 0.3209207332047416 alphas[0] - 0.3775432284760386 alphas[1]
   - 1.035280202557997 alphas[2] + 0.1441054714758315 alphas[3] - alphas[4]
   + f_var[32] = 0
 Q0_upper_bound_32:
   - 1313.714597627199 Fraction_for_second_order_amount[32] + Q0_var[32]
   = 0
 R290: Q0_var[32] + Q_hat_adjusted_32 = 1269.537685221519
 Constr_Left_0_diff_aux_32: Sold_0[32] - Q0_var[32] + Left_0_aux_32 = 0
 Constr_Lost_0_diff_aux_32: Q0_var[32] + Lost_0_aux_32 = 999.7573551464841
 Constr_Q1_plus_left_32: - Left_0[32] + Q1_plus_left[32] - Q1_var[32] = 0
 Constr_Left_1_diff_aux_32: Sold_1[32] - Q1_plus_left[32] + Left_0_aux_32
   = 0
 Constr_Lost_1_diff_aux_32: Q1_plus_left[32] + Lost_0_aux_32
   = 269.7887110625836
 Profit_Constraint_32: - 800 Sold_0[32] + 800 Lost_0[32] - 800 Sold_1[32]
   + 200 Left_1[32] + 800 Lost_1[32] + profits_vars[32] = 0
 R297: 1.039052776544602 alphas[0] + 1.456072968082009 alphas[1]
   - 0.0254054356507122 alphas[2] - 0.0759032806212098 alphas[3]
   - alphas[4] + f_var[33] = 0
 Q0_upper_bound_33:
   - 1313.714597627199 Fraction_for_second_order_amount[33] + Q0_var[33]
   = 0
 R299: Q0_var[33] + Q_hat_adjusted_33 = 1108.972257722064
 Constr_Left_0_diff_aux_33: Sold_0[33] - Q0_var[33] + Left_0_aux_33 = 0
 Constr_Lost_0_diff_aux_33: Q0_var[33] + Lost_0_aux_33 = 895.8880024152131
 Constr_Q1_plus_left_33: - Left_0[33] + Q1_plus_left[33] - Q1_var[33] = 0
 Constr_Left_1_diff_aux_33: Sold_1[33] - Q1_plus_left[33] + Left_0_aux_33
   = 0
 Constr_Lost_1_diff_aux_33: Q1_plus_left[33] + Lost_0_aux_33
   = 213.0842623717018
 Profit_Constraint_33: - 800 Sold_0[33] + 800 Lost_0[33] - 800 Sold_1[33]
   + 200 Left_1[33] + 800 Lost_1[33] + profits_vars[33] = 0
 R306: - 0.6215016003765439 alphas[0] + 0.2910342674354107 alphas[1]
   - 0.1014981941020713 alphas[2] + 0.802583619012584 alphas[3] - alphas[4]
   + f_var[34] = 0
 Q0_upper_bound_34:
   - 1313.714597627199 Fraction_for_second_order_amount[34] + Q0_var[34]
   = 0
 R308: Q0_var[34] + Q_hat_adjusted_34 = 1094.906188129722
 Constr_Left_0_diff_aux_34: Sold_0[34] - Q0_var[34] + Left_0_aux_34 = 0
 Constr_Lost_0_diff_aux_34: Q0_var[34] + Lost_0_aux_34 = 877.5503433512237
 Constr_Q1_plus_left_34: - Left_0[34] + Q1_plus_left[34] - Q1_var[34] = 0
 Constr_Left_1_diff_aux_34: Sold_1[34] - Q1_plus_left[34] + Left_0_aux_34
   = 0
 Constr_Lost_1_diff_aux_34: Q1_plus_left[34] + Lost_0_aux_34
   = 217.3568692916338
 Profit_Constraint_34: - 800 Sold_0[34] + 800 Lost_0[34] - 800 Sold_1[34]
   + 200 Left_1[34] + 800 Lost_1[34] + profits_vars[34] = 0
 R315: 0.4813195019409447 alphas[0] - 1.201832453896129 alphas[1]
   - 0.8080650104156152 alphas[2] + 0.1391758240894182 alphas[3]
   - alphas[4] + f_var[35] = 0
 Q0_upper_bound_35:
   - 1313.714597627199 Fraction_for_second_order_amount[35] + Q0_var[35]
   = 0
 R317: Q0_var[35] + Q_hat_adjusted_35 = 1311.15597599107
 Constr_Left_0_diff_aux_35: Sold_0[35] - Q0_var[35] + Left_0_aux_35 = 0
 Constr_Lost_0_diff_aux_35: Q0_var[35] + Lost_0_aux_35 = 1054.123240031358
 Constr_Q1_plus_left_35: - Left_0[35] + Q1_plus_left[35] - Q1_var[35] = 0
 Constr_Left_1_diff_aux_35: Sold_1[35] - Q1_plus_left[35] + Left_0_aux_35
   = 0
 Constr_Lost_1_diff_aux_35: Q1_plus_left[35] + Lost_0_aux_35
   = 257.0305824046281
 Profit_Constraint_35: - 800 Sold_0[35] + 800 Lost_0[35] - 800 Sold_1[35]
   + 200 Left_1[35] + 800 Lost_1[35] + profits_vars[35] = 0
 R324: 0.1375918242899622 alphas[0] + 0.3762866661996329 alphas[1]
   - 1.437748099382095 alphas[2] + 0.3745986548851619 alphas[3] - alphas[4]
   + f_var[36] = 0
 Q0_upper_bound_36:
   - 1313.714597627199 Fraction_for_second_order_amount[36] + Q0_var[36]
   = 0
 R326: Q0_var[36] + Q_hat_adjusted_36 = 1540.911860152977
 Constr_Left_0_diff_aux_36: Sold_0[36] - Q0_var[36] + Left_0_aux_36 = 0
 Constr_Lost_0_diff_aux_36: Q0_var[36] + Lost_0_aux_36 = 1248.520310356159
 Constr_Q1_plus_left_36: - Left_0[36] + Q1_plus_left[36] - Q1_var[36] = 0
 Constr_Left_1_diff_aux_36: Sold_1[36] - Q1_plus_left[36] + Left_0_aux_36
   = 0
 Constr_Lost_1_diff_aux_36: Q1_plus_left[36] + Lost_0_aux_36
   = 292.3872757094458
 Profit_Constraint_36: - 800 Sold_0[36] + 800 Lost_0[36] - 800 Sold_1[36]
   + 200 Left_1[36] + 800 Lost_1[36] + profits_vars[36] = 0
 R333: - 0.2207470902925119 alphas[0] - 0.6519283605764955 alphas[1]
   - 0.3732484446432362 alphas[2] + 1.492836529817347 alphas[3] - alphas[4]
   + f_var[37] = 0
 Q0_upper_bound_37:
   - 1313.714597627199 Fraction_for_second_order_amount[37] + Q0_var[37]
   = 0
 R335: Q0_var[37] + Q_hat_adjusted_37 = 1186.170340464981
 Constr_Left_0_diff_aux_37: Sold_0[37] - Q0_var[37] + Left_0_aux_37 = 0
 Constr_Lost_0_diff_aux_37: Q0_var[37] + Lost_0_aux_37 = 953.5530525728495
 Constr_Q1_plus_left_37: - Left_0[37] + Q1_plus_left[37] - Q1_var[37] = 0
 Constr_Left_1_diff_aux_37: Sold_1[37] - Q1_plus_left[37] + Left_0_aux_37
   = 0
 Constr_Lost_1_diff_aux_37: Q1_plus_left[37] + Lost_0_aux_37
   = 232.6156406200197
 Profit_Constraint_37: - 800 Sold_0[37] + 800 Lost_0[37] - 800 Sold_1[37]
   + 200 Left_1[37] + 800 Lost_1[37] + profits_vars[37] = 0
 R342: - 0.2485802287940444 alphas[0] - 0.7597209499073978 alphas[1]
   + 0.2169096835646594 alphas[2] - 0.4689274775289497 alphas[3]
   - alphas[4] + f_var[38] = 0
 Q0_upper_bound_38:
   - 1313.714597627199 Fraction_for_second_order_amount[38] + Q0_var[38]
   = 0
 R344: Q0_var[38] + Q_hat_adjusted_38 = 1017.408953251753
 Constr_Left_0_diff_aux_38: Sold_0[38] - Q0_var[38] + Left_0_aux_38 = 0
 Constr_Lost_0_diff_aux_38: Q0_var[38] + Lost_0_aux_38 = 817.9301311827918
 Constr_Q1_plus_left_38: - Left_0[38] + Q1_plus_left[38] - Q1_var[38] = 0
 Constr_Left_1_diff_aux_38: Sold_1[38] - Q1_plus_left[38] + Left_0_aux_38
   = 0
 Constr_Lost_1_diff_aux_38: Q1_plus_left[38] + Lost_0_aux_38
   = 199.4782730997528
 Profit_Constraint_38: - 800 Sold_0[38] + 800 Lost_0[38] - 800 Sold_1[38]
   + 200 Left_1[38] + 800 Lost_1[38] + profits_vars[38] = 0
 R351: - 1.659314904438632 alphas[0] - 0.0242172473466918 alphas[1]
   - 1.49316668011131 alphas[2] - 0.7214739169808944 alphas[3] - alphas[4]
   + f_var[39] = 0
 Q0_upper_bound_39:
   - 1313.714597627199 Fraction_for_second_order_amount[39] + Q0_var[39]
   = 0
 R353: Q0_var[39] + Q_hat_adjusted_39 = 1546.556512830125
 Constr_Left_0_diff_aux_39: Sold_0[39] - Q0_var[39] + Left_0_aux_39 = 0
 Constr_Lost_0_diff_aux_39: Q0_var[39] + Lost_0_aux_39 = 1251.05187943845
 Constr_Q1_plus_left_39: - Left_0[39] + Q1_plus_left[39] - Q1_var[39] = 0
 Constr_Left_1_diff_aux_39: Sold_1[39] - Q1_plus_left[39] + Left_0_aux_39
   = 0
 Constr_Lost_1_diff_aux_39: Q1_plus_left[39] + Lost_0_aux_39
   = 295.4990279488894
 Profit_Constraint_39: - 800 Sold_0[39] + 800 Lost_0[39] - 800 Sold_1[39]
   + 200 Left_1[39] + 800 Lost_1[39] + profits_vars[39] = 0
 R360: 1.449198890678448 alphas[0] - 0.2664943891526906 alphas[1]
   + 0.2846605829059055 alphas[2] + 1.207549412180324 alphas[3] - alphas[4]
   + f_var[40] = 0
 Q0_upper_bound_40:
   - 1313.714597627199 Fraction_for_second_order_amount[40] + Q0_var[40]
   = 0
 R362: Q0_var[40] + Q_hat_adjusted_40 = 991.2965929212278
 Constr_Left_0_diff_aux_40: Sold_0[40] - Q0_var[40] + Left_0_aux_40 = 0
 Constr_Lost_0_diff_aux_40: Q0_var[40] + Lost_0_aux_40 = 795.6238533141709
 Constr_Q1_plus_left_40: - Left_0[40] + Q1_plus_left[40] - Q1_var[40] = 0
 Constr_Left_1_diff_aux_40: Sold_1[40] - Q1_plus_left[40] + Left_0_aux_40
   = 0
 Constr_Lost_1_diff_aux_40: Q1_plus_left[40] + Lost_0_aux_40
   = 195.6740614080023
 Profit_Constraint_40: - 800 Sold_0[40] + 800 Lost_0[40] - 800 Sold_1[40]
   + 200 Left_1[40] + 800 Lost_1[40] + profits_vars[40] = 0
 R369: - 0.3759093595592308 alphas[0] - 0.2116717018428169 alphas[1]
   + 0.6558105828549073 alphas[2] + 0.8504312172138101 alphas[3]
   - alphas[4] + f_var[41] = 0
 Q0_upper_bound_41:
   - 1313.714597627199 Fraction_for_second_order_amount[41] + Q0_var[41]
   = 0
 R371: Q0_var[41] + Q_hat_adjusted_41 = 829.4314623419906
 Constr_Left_0_diff_aux_41: Sold_0[41] - Q0_var[41] + Left_0_aux_41 = 0
 Constr_Lost_0_diff_aux_41: Q0_var[41] + Lost_0_aux_41 = 654.5950569754489
 Constr_Q1_plus_left_41: - Left_0[41] + Q1_plus_left[41] - Q1_var[41] = 0
 Constr_Left_1_diff_aux_41: Sold_1[41] - Q1_plus_left[41] + Left_0_aux_41
   = 0
 Constr_Lost_1_diff_aux_41: Q1_plus_left[41] + Lost_0_aux_41
   = 174.8339960668562
 Profit_Constraint_41: - 800 Sold_0[41] + 800 Lost_0[41] - 800 Sold_1[41]
   + 200 Left_1[41] + 800 Lost_1[41] + profits_vars[41] = 0
 R378: - 1.03480446944358 alphas[0] + 0.0451248125255332 alphas[1]
   - 1.119722262308427 alphas[2] + 1.307290387131822 alphas[3] - alphas[4]
   + f_var[42] = 0
 Q0_upper_bound_42:
   - 1313.714597627199 Fraction_for_second_order_amount[42] + Q0_var[42]
   = 0
 R380: Q0_var[42] + Q_hat_adjusted_42 = 1328.177473651711
 Constr_Left_0_diff_aux_42: Sold_0[42] - Q0_var[42] + Left_0_aux_42 = 0
 Constr_Lost_0_diff_aux_42: Q0_var[42] + Lost_0_aux_42 = 1053.650125320466
 Constr_Q1_plus_left_42: - Left_0[42] + Q1_plus_left[42] - Q1_var[42] = 0
 Constr_Left_1_diff_aux_42: Sold_1[42] - Q1_plus_left[42] + Left_0_aux_42
   = 0
 Constr_Lost_1_diff_aux_42: Q1_plus_left[42] + Lost_0_aux_42
   = 274.5301310894791
 Profit_Constraint_42: - 800 Sold_0[42] + 800 Lost_0[42] - 800 Sold_1[42]
   + 200 Left_1[42] + 800 Lost_1[42] + profits_vars[42] = 0
 R387: - 0.6590207154980874 alphas[0] + 0.9718029380865217 alphas[1]
   + 1.941414310410371 alphas[2] - 0.2791442553148147 alphas[3] - alphas[4]
   + f_var[43] = 0
 Q0_upper_bound_43:
   - 1313.714597627199 Fraction_for_second_order_amount[43] + Q0_var[43]
   = 0
 R389: Q0_var[43] + Q_hat_adjusted_43 = 495.2427505843179
 Constr_Left_0_diff_aux_43: Sold_0[43] - Q0_var[43] + Left_0_aux_43 = 0
 Constr_Lost_0_diff_aux_43: Q0_var[43] + Lost_0_aux_43 = 392.6184481779978
 Constr_Q1_plus_left_43: - Left_0[43] + Q1_plus_left[43] - Q1_var[43] = 0
 Constr_Left_1_diff_aux_43: Sold_1[43] - Q1_plus_left[43] + Left_0_aux_43
   = 0
 Constr_Lost_1_diff_aux_43: Q1_plus_left[43] + Lost_0_aux_43
   = 102.6473715517801
 Profit_Constraint_43: - 800 Sold_0[43] + 800 Lost_0[43] - 800 Sold_1[43]
   + 200 Left_1[43] + 800 Lost_1[43] + profits_vars[43] = 0
 R396: 0.429483820606838 alphas[0] - 1.542860046630717 alphas[1]
   - 1.205281107690666 alphas[2] + 1.735995123752914 alphas[3] - alphas[4]
   + f_var[44] = 0
 Q0_upper_bound_44:
   - 1313.714597627199 Fraction_for_second_order_amount[44] + Q0_var[44]
   = 0
 R398: Q0_var[44] + Q_hat_adjusted_44 = 1372.974390718279
 Constr_Left_0_diff_aux_44: Sold_0[44] - Q0_var[44] + Left_0_aux_44 = 0
 Constr_Lost_0_diff_aux_44: Q0_var[44] + Lost_0_aux_44 = 1093.639865674397
 Constr_Q1_plus_left_44: - Left_0[44] + Q1_plus_left[44] - Q1_var[44] = 0
 Constr_Left_1_diff_aux_44: Sold_1[44] - Q1_plus_left[44] + Left_0_aux_44
   = 0
 Constr_Lost_1_diff_aux_44: Q1_plus_left[44] + Lost_0_aux_44
   = 279.3342586080686
 Profit_Constraint_44: - 800 Sold_0[44] + 800 Lost_0[44] - 800 Sold_1[44]
   + 200 Left_1[44] + 800 Lost_1[44] + profits_vars[44] = 0
 R405: - 0.2857986047027599 alphas[0] + 1.008662274909803 alphas[1]
   + 1.640051333429476 alphas[2] + 0.5078173339992713 alphas[3] - alphas[4]
   + f_var[45] = 0
 Q0_upper_bound_45:
   - 1313.714597627199 Fraction_for_second_order_amount[45] + Q0_var[45]
   = 0
 R407: Q0_var[45] + Q_hat_adjusted_45 = 577.8131869823724
 Constr_Left_0_diff_aux_45: Sold_0[45] - Q0_var[45] + Left_0_aux_45 = 0
 Constr_Lost_0_diff_aux_45: Q0_var[45] + Lost_0_aux_45 = 458.2379216265915
 Constr_Q1_plus_left_45: - Left_0[45] + Q1_plus_left[45] - Q1_var[45] = 0
 Constr_Left_1_diff_aux_45: Sold_1[45] - Q1_plus_left[45] + Left_0_aux_45
   = 0
 Constr_Lost_1_diff_aux_45: Q1_plus_left[45] + Lost_0_aux_45
   = 119.5688968988028
 Profit_Constraint_45: - 800 Sold_0[45] + 800 Lost_0[45] - 800 Sold_1[45]
   + 200 Left_1[45] + 800 Lost_1[45] + profits_vars[45] = 0
 R414: 1.438065666099115 alphas[0] + 0.5423413959653262 alphas[1]
   - 0.1001654054989482 alphas[2] + 0.3519247535065088 alphas[3]
   - alphas[4] + f_var[46] = 0
 Q0_upper_bound_46:
   - 1313.714597627199 Fraction_for_second_order_amount[46] + Q0_var[46]
   = 0
 R416: Q0_var[46] + Q_hat_adjusted_46 = 1210.250054587914
 Constr_Left_0_diff_aux_46: Sold_0[46] - Q0_var[46] + Left_0_aux_46 = 0
 Constr_Lost_0_diff_aux_46: Q0_var[46] + Lost_0_aux_46 = 992.967748356878
 Constr_Q1_plus_left_46: - Left_0[46] + Q1_plus_left[46] - Q1_var[46] = 0
 Constr_Left_1_diff_aux_46: Sold_1[46] - Q1_plus_left[46] + Left_0_aux_46
   = 0
 Constr_Lost_1_diff_aux_46: Q1_plus_left[46] + Lost_0_aux_46
   = 217.2820332372652
 Profit_Constraint_46: - 800 Sold_0[46] + 800 Lost_0[46] - 800 Sold_1[46]
   + 200 Left_1[46] + 800 Lost_1[46] + profits_vars[46] = 0
 R423: 1.087661489121398 alphas[0] - 1.622967272711107 alphas[1]
   - 0.3976058652627706 alphas[2] - 1.372873859089517 alphas[3] - alphas[4]
   + f_var[47] = 0
 Q0_upper_bound_47:
   - 1313.714597627199 Fraction_for_second_order_amount[47] + Q0_var[47]
   = 0
 R425: Q0_var[47] + Q_hat_adjusted_47 = 1145.784759343474
 Constr_Left_0_diff_aux_47: Sold_0[47] - Q0_var[47] + Left_0_aux_47 = 0
 Constr_Lost_0_diff_aux_47: Q0_var[47] + Lost_0_aux_47 = 776.313105195524
 Constr_Q1_plus_left_47: - Left_0[47] + Q1_plus_left[47] - Q1_var[47] = 0
 Constr_Left_1_diff_aux_47: Sold_1[47] - Q1_plus_left[47] + Left_0_aux_47
   = 0
 Constr_Lost_1_diff_aux_47: Q1_plus_left[47] + Lost_0_aux_47
   = 309.8808319634448
 Profit_Constraint_47: - 800 Sold_0[47] + 800 Lost_0[47] - 800 Sold_1[47]
   + 200 Left_1[47] + 800 Lost_1[47] + profits_vars[47] = 0
 R432: 1.325967768359569 alphas[0] - 1.425838217252366 alphas[1]
   + 1.33574009191027 alphas[2] + 0.6812154148658552 alphas[3] - alphas[4]
   + f_var[48] = 0
 Q0_upper_bound_48:
   - 1313.714597627199 Fraction_for_second_order_amount[48] + Q0_var[48]
   = 0
 R434: Q0_var[48] + Q_hat_adjusted_48 = 675.1953373235899
 Constr_Left_0_diff_aux_48: Sold_0[48] - Q0_var[48] + Left_0_aux_48 = 0
 Constr_Lost_0_diff_aux_48: Q0_var[48] + Lost_0_aux_48 = 467.383717899614
 Constr_Q1_plus_left_48: - Left_0[48] + Q1_plus_left[48] - Q1_var[48] = 0
 Constr_Left_1_diff_aux_48: Sold_1[48] - Q1_plus_left[48] + Left_0_aux_48
   = 0
 Constr_Lost_1_diff_aux_48: Q1_plus_left[48] + Lost_0_aux_48
   = 212.4098433917106
 Profit_Constraint_48: - 800 Sold_0[48] + 800 Lost_0[48] - 800 Sold_1[48]
   + 200 Left_1[48] + 800 Lost_1[48] + profits_vars[48] = 0
 R441: - 0.5684780678612856 alphas[0] - 1.027320704153287 alphas[1]
   - 0.5648121011339805 alphas[2] - 1.079183913932596 alphas[3] - alphas[4]
   + f_var[49] = 0
 Q0_upper_bound_49:
   - 1313.714597627199 Fraction_for_second_order_amount[49] + Q0_var[49]
   = 0
 R443: Q0_var[49] + Q_hat_adjusted_49 = 1227.58002344265
 Constr_Left_0_diff_aux_49: Sold_0[49] - Q0_var[49] + Left_0_aux_49 = 0
 Constr_Lost_0_diff_aux_49: Q0_var[49] + Lost_0_aux_49 = 844.7483032693539
 Constr_Q1_plus_left_49: - Left_0[49] + Q1_plus_left[49] - Q1_var[49] = 0
 Constr_Left_1_diff_aux_49: Sold_1[49] - Q1_plus_left[49] + Left_0_aux_49
   = 0
 Constr_Lost_1_diff_aux_49: Q1_plus_left[49] + Lost_0_aux_49
   = 390.7725466006844
 Profit_Constraint_49: - 800 Sold_0[49] + 800 Lost_0[49] - 800 Sold_1[49]
   + 200 Left_1[49] + 800 Lost_1[49] + profits_vars[49] = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
 alphas[4] free
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
 Fraction_for_second_order_amount[0] <= 1
 Fraction_for_second_order_amount[1] <= 1
 Fraction_for_second_order_amount[2] <= 1
 Fraction_for_second_order_amount[3] <= 1
 Fraction_for_second_order_amount[4] <= 1
 Fraction_for_second_order_amount[5] <= 1
 Fraction_for_second_order_amount[6] <= 1
 Fraction_for_second_order_amount[7] <= 1
 Fraction_for_second_order_amount[8] <= 1
 Fraction_for_second_order_amount[9] <= 1
 Fraction_for_second_order_amount[10] <= 1
 Fraction_for_second_order_amount[11] <= 1
 Fraction_for_second_order_amount[12] <= 1
 Fraction_for_second_order_amount[13] <= 1
 Fraction_for_second_order_amount[14] <= 1
 Fraction_for_second_order_amount[15] <= 1
 Fraction_for_second_order_amount[16] <= 1
 Fraction_for_second_order_amount[17] <= 1
 Fraction_for_second_order_amount[18] <= 1
 Fraction_for_second_order_amount[19] <= 1
 Fraction_for_second_order_amount[20] <= 1
 Fraction_for_second_order_amount[21] <= 1
 Fraction_for_second_order_amount[22] <= 1
 Fraction_for_second_order_amount[23] <= 1
 Fraction_for_second_order_amount[24] <= 1
 Fraction_for_second_order_amount[25] <= 1
 Fraction_for_second_order_amount[26] <= 1
 Fraction_for_second_order_amount[27] <= 1
 Fraction_for_second_order_amount[28] <= 1
 Fraction_for_second_order_amount[29] <= 1
 Fraction_for_second_order_amount[30] <= 1
 Fraction_for_second_order_amount[31] <= 1
 Fraction_for_second_order_amount[32] <= 1
 Fraction_for_second_order_amount[33] <= 1
 Fraction_for_second_order_amount[34] <= 1
 Fraction_for_second_order_amount[35] <= 1
 Fraction_for_second_order_amount[36] <= 1
 Fraction_for_second_order_amount[37] <= 1
 Fraction_for_second_order_amount[38] <= 1
 Fraction_for_second_order_amount[39] <= 1
 Fraction_for_second_order_amount[40] <= 1
 Fraction_for_second_order_amount[41] <= 1
 Fraction_for_second_order_amount[42] <= 1
 Fraction_for_second_order_amount[43] <= 1
 Fraction_for_second_order_amount[44] <= 1
 Fraction_for_second_order_amount[45] <= 1
 Fraction_for_second_order_amount[46] <= 1
 Fraction_for_second_order_amount[47] <= 1
 Fraction_for_second_order_amount[48] <= 1
 Fraction_for_second_order_amount[49] <= 1
 Q0_var[0] <= 1314.714597627199
 Q0_var[1] <= 1314.714597627199
 Q0_var[2] <= 1314.714597627199
 Q0_var[3] <= 1314.714597627199
 Q0_var[4] <= 1314.714597627199
 Q0_var[5] <= 1314.714597627199
 Q0_var[6] <= 1314.714597627199
 Q0_var[7] <= 1314.714597627199
 Q0_var[8] <= 1314.714597627199
 Q0_var[9] <= 1314.714597627199
 Q0_var[10] <= 1314.714597627199
 Q0_var[11] <= 1314.714597627199
 Q0_var[12] <= 1314.714597627199
 Q0_var[13] <= 1314.714597627199
 Q0_var[14] <= 1314.714597627199
 Q0_var[15] <= 1314.714597627199
 Q0_var[16] <= 1314.714597627199
 Q0_var[17] <= 1314.714597627199
 Q0_var[18] <= 1314.714597627199
 Q0_var[19] <= 1314.714597627199
 Q0_var[20] <= 1314.714597627199
 Q0_var[21] <= 1314.714597627199
 Q0_var[22] <= 1314.714597627199
 Q0_var[23] <= 1314.714597627199
 Q0_var[24] <= 1314.714597627199
 Q0_var[25] <= 1314.714597627199
 Q0_var[26] <= 1314.714597627199
 Q0_var[27] <= 1314.714597627199
 Q0_var[28] <= 1314.714597627199
 Q0_var[29] <= 1314.714597627199
 Q0_var[30] <= 1314.714597627199
 Q0_var[31] <= 1314.714597627199
 Q0_var[32] <= 1314.714597627199
 Q0_var[33] <= 1314.714597627199
 Q0_var[34] <= 1314.714597627199
 Q0_var[35] <= 1314.714597627199
 Q0_var[36] <= 1314.714597627199
 Q0_var[37] <= 1314.714597627199
 Q0_var[38] <= 1314.714597627199
 Q0_var[39] <= 1314.714597627199
 Q0_var[40] <= 1314.714597627199
 Q0_var[41] <= 1314.714597627199
 Q0_var[42] <= 1314.714597627199
 Q0_var[43] <= 1314.714597627199
 Q0_var[44] <= 1314.714597627199
 Q0_var[45] <= 1314.714597627199
 Q0_var[46] <= 1314.714597627199
 Q0_var[47] <= 1314.714597627199
 Q0_var[48] <= 1314.714597627199
 Q0_var[49] <= 1314.714597627199
 Q_hat_adjusted_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Q_hat_adjusted_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Q_hat_adjusted_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Q_hat_adjusted_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Q_hat_adjusted_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Q_hat_adjusted_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Q_hat_adjusted_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Q_hat_adjusted_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Q_hat_adjusted_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Q_hat_adjusted_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Q_hat_adjusted_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Q_hat_adjusted_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Q_hat_adjusted_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Q_hat_adjusted_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Q_hat_adjusted_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Q_hat_adjusted_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Q_hat_adjusted_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Q_hat_adjusted_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Q_hat_adjusted_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Q_hat_adjusted_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Q_hat_adjusted_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Q_hat_adjusted_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Q_hat_adjusted_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Q_hat_adjusted_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Q_hat_adjusted_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Q_hat_adjusted_25 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Q_hat_adjusted_26 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Q_hat_adjusted_27 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Q_hat_adjusted_28 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Q_hat_adjusted_29 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Q_hat_adjusted_30 free
 Left_0_aux_30 free
 Lost_0_aux_30 free
 Left_0_aux_30 free
 Lost_0_aux_30 free
 Q_hat_adjusted_31 free
 Left_0_aux_31 free
 Lost_0_aux_31 free
 Left_0_aux_31 free
 Lost_0_aux_31 free
 Q_hat_adjusted_32 free
 Left_0_aux_32 free
 Lost_0_aux_32 free
 Left_0_aux_32 free
 Lost_0_aux_32 free
 Q_hat_adjusted_33 free
 Left_0_aux_33 free
 Lost_0_aux_33 free
 Left_0_aux_33 free
 Lost_0_aux_33 free
 Q_hat_adjusted_34 free
 Left_0_aux_34 free
 Lost_0_aux_34 free
 Left_0_aux_34 free
 Lost_0_aux_34 free
 Q_hat_adjusted_35 free
 Left_0_aux_35 free
 Lost_0_aux_35 free
 Left_0_aux_35 free
 Lost_0_aux_35 free
 Q_hat_adjusted_36 free
 Left_0_aux_36 free
 Lost_0_aux_36 free
 Left_0_aux_36 free
 Lost_0_aux_36 free
 Q_hat_adjusted_37 free
 Left_0_aux_37 free
 Lost_0_aux_37 free
 Left_0_aux_37 free
 Lost_0_aux_37 free
 Q_hat_adjusted_38 free
 Left_0_aux_38 free
 Lost_0_aux_38 free
 Left_0_aux_38 free
 Lost_0_aux_38 free
 Q_hat_adjusted_39 free
 Left_0_aux_39 free
 Lost_0_aux_39 free
 Left_0_aux_39 free
 Lost_0_aux_39 free
 Q_hat_adjusted_40 free
 Left_0_aux_40 free
 Lost_0_aux_40 free
 Left_0_aux_40 free
 Lost_0_aux_40 free
 Q_hat_adjusted_41 free
 Left_0_aux_41 free
 Lost_0_aux_41 free
 Left_0_aux_41 free
 Lost_0_aux_41 free
 Q_hat_adjusted_42 free
 Left_0_aux_42 free
 Lost_0_aux_42 free
 Left_0_aux_42 free
 Lost_0_aux_42 free
 Q_hat_adjusted_43 free
 Left_0_aux_43 free
 Lost_0_aux_43 free
 Left_0_aux_43 free
 Lost_0_aux_43 free
 Q_hat_adjusted_44 free
 Left_0_aux_44 free
 Lost_0_aux_44 free
 Left_0_aux_44 free
 Lost_0_aux_44 free
 Q_hat_adjusted_45 free
 Left_0_aux_45 free
 Lost_0_aux_45 free
 Left_0_aux_45 free
 Lost_0_aux_45 free
 Q_hat_adjusted_46 free
 Left_0_aux_46 free
 Lost_0_aux_46 free
 Left_0_aux_46 free
 Lost_0_aux_46 free
 Q_hat_adjusted_47 free
 Left_0_aux_47 free
 Lost_0_aux_47 free
 Left_0_aux_47 free
 Lost_0_aux_47 free
 Q_hat_adjusted_48 free
 Left_0_aux_48 free
 Lost_0_aux_48 free
 Left_0_aux_48 free
 Lost_0_aux_48 free
 Q_hat_adjusted_49 free
 Left_0_aux_49 free
 Lost_0_aux_49 free
 Left_0_aux_49 free
 Lost_0_aux_49 free
General Constraints
 logistic_constraint_0: Fraction_for_second_order_amount[0] =
   LOGISTIC ( f_var[0] )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 455.7734011299941 )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Q1_plus_left[0] , 127.3800543371205 )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 logistic_constraint_1: Fraction_for_second_order_amount[1] =
   LOGISTIC ( f_var[1] )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 433.4550214245154 )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Q1_plus_left[1] , 350.703854210086 )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 logistic_constraint_2: Fraction_for_second_order_amount[2] =
   LOGISTIC ( f_var[2] )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 610.1860591594664 )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Q1_plus_left[2] , 162.3591764004619 )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 logistic_constraint_3: Fraction_for_second_order_amount[3] =
   LOGISTIC ( f_var[3] )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 169.823119790778 )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Q1_plus_left[3] , 575.7356520491599 )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 logistic_constraint_4: Fraction_for_second_order_amount[4] =
   LOGISTIC ( f_var[4] )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1195.540081262036 )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Q1_plus_left[4] , 289.4741180977849 )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 logistic_constraint_5: Fraction_for_second_order_amount[5] =
   LOGISTIC ( f_var[5] )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 625.3860526814287 )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Q1_plus_left[5] , 171.1225475699914 )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 logistic_constraint_6: Fraction_for_second_order_amount[6] =
   LOGISTIC ( f_var[6] )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 575.5844350405919 )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Q1_plus_left[6] , 205.4102025365846 )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 logistic_constraint_7: Fraction_for_second_order_amount[7] =
   LOGISTIC ( f_var[7] )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 680.6800123025196 )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Q1_plus_left[7] , 184.9370937503554 )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 logistic_constraint_8: Fraction_for_second_order_amount[8] =
   LOGISTIC ( f_var[8] )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1083.2898673318 )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Q1_plus_left[8] , 259.6093667825127 )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 logistic_constraint_9: Fraction_for_second_order_amount[9] =
   LOGISTIC ( f_var[9] )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 1107.09836748275 )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Q1_plus_left[9] , 253.0650507613931 )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 logistic_constraint_10: Fraction_for_second_order_amount[10] =
   LOGISTIC ( f_var[10] )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 1310.987616267015 )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Q1_plus_left[10]
   , 298.0677894793409 )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 logistic_constraint_11: Fraction_for_second_order_amount[11] =
   LOGISTIC ( f_var[11] )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 608.7737460707782 )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Q1_plus_left[11]
   , 142.6623954734695 )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 logistic_constraint_12: Fraction_for_second_order_amount[12] =
   LOGISTIC ( f_var[12] )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 621.2755875215321 )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Q1_plus_left[12]
   , 156.3460211507899 )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 logistic_constraint_13: Fraction_for_second_order_amount[13] =
   LOGISTIC ( f_var[13] )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 902.339313510209 )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Q1_plus_left[13]
   , 216.8952137911535 )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 logistic_constraint_14: Fraction_for_second_order_amount[14] =
   LOGISTIC ( f_var[14] )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 1168.030156567468 )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Q1_plus_left[14]
   , 274.8575933249894 )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
 logistic_constraint_15: Fraction_for_second_order_amount[15] =
   LOGISTIC ( f_var[15] )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 446.9623868488507 )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Q1_plus_left[15]
   , 227.7165654975522 )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_0_aux_15 , 0 )
 logistic_constraint_16: Fraction_for_second_order_amount[16] =
   LOGISTIC ( f_var[16] )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 806.6113964906422 )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Q1_plus_left[16]
   , 203.6401427861327 )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_0_aux_16 , 0 )
 logistic_constraint_17: Fraction_for_second_order_amount[17] =
   LOGISTIC ( f_var[17] )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 586.5977877510986 )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Q1_plus_left[17]
   , 236.436415247552 )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_0_aux_17 , 0 )
 logistic_constraint_18: Fraction_for_second_order_amount[18] =
   LOGISTIC ( f_var[18] )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 386.2125767106902 )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Q1_plus_left[18]
   , 104.1301995882807 )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_0_aux_18 , 0 )
 logistic_constraint_19: Fraction_for_second_order_amount[19] =
   LOGISTIC ( f_var[19] )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 857.0158397214361 )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Q1_plus_left[19]
   , 227.8945032797447 )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_0_aux_19 , 0 )
 logistic_constraint_20: Fraction_for_second_order_amount[20] =
   LOGISTIC ( f_var[20] )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 1040.317914110936 )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Q1_plus_left[20]
   , 266.209691047238 )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_0_aux_20 , 0 )
 logistic_constraint_21: Fraction_for_second_order_amount[21] =
   LOGISTIC ( f_var[21] )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 850.8794252886834 )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Q1_plus_left[21]
   , 246.1711613540316 )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_0_aux_21 , 0 )
 logistic_constraint_22: Fraction_for_second_order_amount[22] =
   LOGISTIC ( f_var[22] )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 648.0700582895297 )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Q1_plus_left[22]
   , 278.6687724412353 )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_0_aux_22 , 0 )
 logistic_constraint_23: Fraction_for_second_order_amount[23] =
   LOGISTIC ( f_var[23] )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1062.368340656185 )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Q1_plus_left[23]
   , 272.4383034843366 )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_0_aux_23 , 0 )
 logistic_constraint_24: Fraction_for_second_order_amount[24] =
   LOGISTIC ( f_var[24] )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1220.801283411015 )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Q1_plus_left[24]
   , 294.5838978046261 )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_0_aux_24 , 0 )
 logistic_constraint_25: Fraction_for_second_order_amount[25] =
   LOGISTIC ( f_var[25] )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 839.8568396032224 )
 Constr_Left_0_max_25: Left_0[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_0_max_25: Lost_0[25] = MAX ( Lost_0_aux_25 , 0 )
 Constr_Sold_1_min_25: Sold_1[25] = MIN ( Q1_plus_left[25]
   , 197.1255191869246 )
 Constr_Left_1_max_25: Left_1[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_1_max_25: Lost_1[25] = MAX ( Lost_0_aux_25 , 0 )
 logistic_constraint_26: Fraction_for_second_order_amount[26] =
   LOGISTIC ( f_var[26] )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 750.3761583847518 )
 Constr_Left_0_max_26: Left_0[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_0_max_26: Lost_0[26] = MAX ( Lost_0_aux_26 , 0 )
 Constr_Sold_1_min_26: Sold_1[26] = MIN ( Q1_plus_left[26]
   , 323.8468923707702 )
 Constr_Left_1_max_26: Left_1[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_1_max_26: Lost_1[26] = MAX ( Lost_0_aux_26 , 0 )
 logistic_constraint_27: Fraction_for_second_order_amount[27] =
   LOGISTIC ( f_var[27] )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 795.999852778667 )
 Constr_Left_0_max_27: Left_0[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_0_max_27: Lost_0[27] = MAX ( Lost_0_aux_27 , 0 )
 Constr_Sold_1_min_27: Sold_1[27] = MIN ( Q1_plus_left[27]
   , 202.2637965092961 )
 Constr_Left_1_max_27: Left_1[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_1_max_27: Lost_1[27] = MAX ( Lost_0_aux_27 , 0 )
 logistic_constraint_28: Fraction_for_second_order_amount[28] =
   LOGISTIC ( f_var[28] )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 672.9453731245621 )
 Constr_Left_0_max_28: Left_0[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_0_max_28: Lost_0[28] = MAX ( Lost_0_aux_28 , 0 )
 Constr_Sold_1_min_28: Sold_1[28] = MIN ( Q1_plus_left[28]
   , 167.631790367369 )
 Constr_Left_1_max_28: Left_1[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_1_max_28: Lost_1[28] = MAX ( Lost_0_aux_28 , 0 )
 logistic_constraint_29: Fraction_for_second_order_amount[29] =
   LOGISTIC ( f_var[29] )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 824.1541446040673 )
 Constr_Left_0_max_29: Left_0[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_0_max_29: Lost_0[29] = MAX ( Lost_0_aux_29 , 0 )
 Constr_Sold_1_min_29: Sold_1[29] = MIN ( Q1_plus_left[29]
   , 528.2279131654524 )
 Constr_Left_1_max_29: Left_1[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_1_max_29: Lost_1[29] = MAX ( Lost_0_aux_29 , 0 )
 logistic_constraint_30: Fraction_for_second_order_amount[30] =
   LOGISTIC ( f_var[30] )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 1157.522389180815 )
 Constr_Left_0_max_30: Left_0[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_0_max_30: Lost_0[30] = MAX ( Lost_0_aux_30 , 0 )
 Constr_Sold_1_min_30: Sold_1[30] = MIN ( Q1_plus_left[30]
   , 290.5583313943889 )
 Constr_Left_1_max_30: Left_1[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_1_max_30: Lost_1[30] = MAX ( Lost_0_aux_30 , 0 )
 logistic_constraint_31: Fraction_for_second_order_amount[31] =
   LOGISTIC ( f_var[31] )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 738.2604881943967 )
 Constr_Left_0_max_31: Left_0[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_0_max_31: Lost_0[31] = MAX ( Lost_0_aux_31 , 0 )
 Constr_Sold_1_min_31: Sold_1[31] = MIN ( Q1_plus_left[31]
   , 192.7150848729526 )
 Constr_Left_1_max_31: Left_1[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_1_max_31: Lost_1[31] = MAX ( Lost_0_aux_31 , 0 )
 logistic_constraint_32: Fraction_for_second_order_amount[32] =
   LOGISTIC ( f_var[32] )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 999.7573551464841 )
 Constr_Left_0_max_32: Left_0[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_0_max_32: Lost_0[32] = MAX ( Lost_0_aux_32 , 0 )
 Constr_Sold_1_min_32: Sold_1[32] = MIN ( Q1_plus_left[32]
   , 269.7887110625836 )
 Constr_Left_1_max_32: Left_1[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_1_max_32: Lost_1[32] = MAX ( Lost_0_aux_32 , 0 )
 logistic_constraint_33: Fraction_for_second_order_amount[33] =
   LOGISTIC ( f_var[33] )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 895.8880024152131 )
 Constr_Left_0_max_33: Left_0[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_0_max_33: Lost_0[33] = MAX ( Lost_0_aux_33 , 0 )
 Constr_Sold_1_min_33: Sold_1[33] = MIN ( Q1_plus_left[33]
   , 213.0842623717018 )
 Constr_Left_1_max_33: Left_1[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_1_max_33: Lost_1[33] = MAX ( Lost_0_aux_33 , 0 )
 logistic_constraint_34: Fraction_for_second_order_amount[34] =
   LOGISTIC ( f_var[34] )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 877.5503433512237 )
 Constr_Left_0_max_34: Left_0[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_0_max_34: Lost_0[34] = MAX ( Lost_0_aux_34 , 0 )
 Constr_Sold_1_min_34: Sold_1[34] = MIN ( Q1_plus_left[34]
   , 217.3568692916338 )
 Constr_Left_1_max_34: Left_1[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_1_max_34: Lost_1[34] = MAX ( Lost_0_aux_34 , 0 )
 logistic_constraint_35: Fraction_for_second_order_amount[35] =
   LOGISTIC ( f_var[35] )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 1054.123240031358 )
 Constr_Left_0_max_35: Left_0[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_0_max_35: Lost_0[35] = MAX ( Lost_0_aux_35 , 0 )
 Constr_Sold_1_min_35: Sold_1[35] = MIN ( Q1_plus_left[35]
   , 257.0305824046281 )
 Constr_Left_1_max_35: Left_1[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_1_max_35: Lost_1[35] = MAX ( Lost_0_aux_35 , 0 )
 logistic_constraint_36: Fraction_for_second_order_amount[36] =
   LOGISTIC ( f_var[36] )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 1248.520310356159 )
 Constr_Left_0_max_36: Left_0[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_0_max_36: Lost_0[36] = MAX ( Lost_0_aux_36 , 0 )
 Constr_Sold_1_min_36: Sold_1[36] = MIN ( Q1_plus_left[36]
   , 292.3872757094458 )
 Constr_Left_1_max_36: Left_1[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_1_max_36: Lost_1[36] = MAX ( Lost_0_aux_36 , 0 )
 logistic_constraint_37: Fraction_for_second_order_amount[37] =
   LOGISTIC ( f_var[37] )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 953.5530525728495 )
 Constr_Left_0_max_37: Left_0[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_0_max_37: Lost_0[37] = MAX ( Lost_0_aux_37 , 0 )
 Constr_Sold_1_min_37: Sold_1[37] = MIN ( Q1_plus_left[37]
   , 232.6156406200197 )
 Constr_Left_1_max_37: Left_1[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_1_max_37: Lost_1[37] = MAX ( Lost_0_aux_37 , 0 )
 logistic_constraint_38: Fraction_for_second_order_amount[38] =
   LOGISTIC ( f_var[38] )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 817.9301311827918 )
 Constr_Left_0_max_38: Left_0[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_0_max_38: Lost_0[38] = MAX ( Lost_0_aux_38 , 0 )
 Constr_Sold_1_min_38: Sold_1[38] = MIN ( Q1_plus_left[38]
   , 199.4782730997528 )
 Constr_Left_1_max_38: Left_1[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_1_max_38: Lost_1[38] = MAX ( Lost_0_aux_38 , 0 )
 logistic_constraint_39: Fraction_for_second_order_amount[39] =
   LOGISTIC ( f_var[39] )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 1251.05187943845 )
 Constr_Left_0_max_39: Left_0[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_0_max_39: Lost_0[39] = MAX ( Lost_0_aux_39 , 0 )
 Constr_Sold_1_min_39: Sold_1[39] = MIN ( Q1_plus_left[39]
   , 295.4990279488894 )
 Constr_Left_1_max_39: Left_1[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_1_max_39: Lost_1[39] = MAX ( Lost_0_aux_39 , 0 )
 logistic_constraint_40: Fraction_for_second_order_amount[40] =
   LOGISTIC ( f_var[40] )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 795.6238533141709 )
 Constr_Left_0_max_40: Left_0[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_0_max_40: Lost_0[40] = MAX ( Lost_0_aux_40 , 0 )
 Constr_Sold_1_min_40: Sold_1[40] = MIN ( Q1_plus_left[40]
   , 195.6740614080023 )
 Constr_Left_1_max_40: Left_1[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_1_max_40: Lost_1[40] = MAX ( Lost_0_aux_40 , 0 )
 logistic_constraint_41: Fraction_for_second_order_amount[41] =
   LOGISTIC ( f_var[41] )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 654.5950569754489 )
 Constr_Left_0_max_41: Left_0[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_0_max_41: Lost_0[41] = MAX ( Lost_0_aux_41 , 0 )
 Constr_Sold_1_min_41: Sold_1[41] = MIN ( Q1_plus_left[41]
   , 174.8339960668562 )
 Constr_Left_1_max_41: Left_1[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_1_max_41: Lost_1[41] = MAX ( Lost_0_aux_41 , 0 )
 logistic_constraint_42: Fraction_for_second_order_amount[42] =
   LOGISTIC ( f_var[42] )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 1053.650125320466 )
 Constr_Left_0_max_42: Left_0[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_0_max_42: Lost_0[42] = MAX ( Lost_0_aux_42 , 0 )
 Constr_Sold_1_min_42: Sold_1[42] = MIN ( Q1_plus_left[42]
   , 274.5301310894791 )
 Constr_Left_1_max_42: Left_1[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_1_max_42: Lost_1[42] = MAX ( Lost_0_aux_42 , 0 )
 logistic_constraint_43: Fraction_for_second_order_amount[43] =
   LOGISTIC ( f_var[43] )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 392.6184481779978 )
 Constr_Left_0_max_43: Left_0[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_0_max_43: Lost_0[43] = MAX ( Lost_0_aux_43 , 0 )
 Constr_Sold_1_min_43: Sold_1[43] = MIN ( Q1_plus_left[43]
   , 102.6473715517801 )
 Constr_Left_1_max_43: Left_1[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_1_max_43: Lost_1[43] = MAX ( Lost_0_aux_43 , 0 )
 logistic_constraint_44: Fraction_for_second_order_amount[44] =
   LOGISTIC ( f_var[44] )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 1093.639865674397 )
 Constr_Left_0_max_44: Left_0[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_0_max_44: Lost_0[44] = MAX ( Lost_0_aux_44 , 0 )
 Constr_Sold_1_min_44: Sold_1[44] = MIN ( Q1_plus_left[44]
   , 279.3342586080686 )
 Constr_Left_1_max_44: Left_1[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_1_max_44: Lost_1[44] = MAX ( Lost_0_aux_44 , 0 )
 logistic_constraint_45: Fraction_for_second_order_amount[45] =
   LOGISTIC ( f_var[45] )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 458.2379216265915 )
 Constr_Left_0_max_45: Left_0[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_0_max_45: Lost_0[45] = MAX ( Lost_0_aux_45 , 0 )
 Constr_Sold_1_min_45: Sold_1[45] = MIN ( Q1_plus_left[45]
   , 119.5688968988028 )
 Constr_Left_1_max_45: Left_1[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_1_max_45: Lost_1[45] = MAX ( Lost_0_aux_45 , 0 )
 logistic_constraint_46: Fraction_for_second_order_amount[46] =
   LOGISTIC ( f_var[46] )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 992.967748356878 )
 Constr_Left_0_max_46: Left_0[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_0_max_46: Lost_0[46] = MAX ( Lost_0_aux_46 , 0 )
 Constr_Sold_1_min_46: Sold_1[46] = MIN ( Q1_plus_left[46]
   , 217.2820332372652 )
 Constr_Left_1_max_46: Left_1[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_1_max_46: Lost_1[46] = MAX ( Lost_0_aux_46 , 0 )
 logistic_constraint_47: Fraction_for_second_order_amount[47] =
   LOGISTIC ( f_var[47] )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 776.313105195524 )
 Constr_Left_0_max_47: Left_0[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_0_max_47: Lost_0[47] = MAX ( Lost_0_aux_47 , 0 )
 Constr_Sold_1_min_47: Sold_1[47] = MIN ( Q1_plus_left[47]
   , 309.8808319634448 )
 Constr_Left_1_max_47: Left_1[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_1_max_47: Lost_1[47] = MAX ( Lost_0_aux_47 , 0 )
 logistic_constraint_48: Fraction_for_second_order_amount[48] =
   LOGISTIC ( f_var[48] )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 467.383717899614 )
 Constr_Left_0_max_48: Left_0[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_0_max_48: Lost_0[48] = MAX ( Lost_0_aux_48 , 0 )
 Constr_Sold_1_min_48: Sold_1[48] = MIN ( Q1_plus_left[48]
   , 212.4098433917106 )
 Constr_Left_1_max_48: Left_1[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_1_max_48: Lost_1[48] = MAX ( Lost_0_aux_48 , 0 )
 logistic_constraint_49: Fraction_for_second_order_amount[49] =
   LOGISTIC ( f_var[49] )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 844.7483032693539 )
 Constr_Left_0_max_49: Left_0[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_0_max_49: Lost_0[49] = MAX ( Lost_0_aux_49 , 0 )
 Constr_Sold_1_min_49: Sold_1[49] = MIN ( Q1_plus_left[49]
   , 390.7725466006844 )
 Constr_Left_1_max_49: Left_1[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_1_max_49: Lost_1[49] = MAX ( Lost_0_aux_49 , 0 )
End
