\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.1 abs_alpha[0] - 0.1 abs_alpha[1] - 0.1 abs_alpha[2]
   - 0.1 abs_alpha[3] + profits_vars[0] + profits_vars[1] + profits_vars[2]
   + profits_vars[3] + profits_vars[4] + profits_vars[5] + profits_vars[6]
   + profits_vars[7] + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14] + profits_vars[15] + profits_vars[16]
   + profits_vars[17] + profits_vars[18] + profits_vars[19]
   + profits_vars[20] + profits_vars[21] + profits_vars[22]
   + profits_vars[23] + profits_vars[24] + profits_vars[25]
   + profits_vars[26] + profits_vars[27] + profits_vars[28]
   + profits_vars[29]
Subject To
 R0: - alphas[0] + abs_alpha[0] >= 0
 R1: alphas[0] + abs_alpha[0] >= 0
 R2: - alphas[1] + abs_alpha[1] >= 0
 R3: alphas[1] + abs_alpha[1] >= 0
 R4: - alphas[2] + abs_alpha[2] >= 0
 R5: alphas[2] + abs_alpha[2] >= 0
 R6: - alphas[3] + abs_alpha[3] >= 0
 R7: alphas[3] + abs_alpha[3] >= 0
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R10: - 19.53277749610589 alphas[0] - alphas[1]
   - 61.81150961736903 alphas[2] - alphas[3] + f_var[0] = 0
 R11: - 2834.593301140667 Fraction[0] + Q0_var[0] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: Q0_var[0] + Lost_0_aux_0 = 554.5300708824453
 R14: Q0_var[0] + Q_hat_adjusted_0 = 616.3413207925975
 Constr_Q1_plus_left_0: - Left_0[0] - Q1_var[0] + Q1_plus_left[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: Q1_plus_left[0] + Lost_0_aux_0
   = 61.81150966750985
 Profit_Constraint_0: - 600 Sold_0[0] + 600 Lost_0[0] - 600 Sold_1[0]
   + 400 Left_1[0] + 600 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R21: - 18.87023666349319 alphas[0] - alphas[1]
   - 67.88545512306519 alphas[2] - alphas[3] + f_var[1] = 0
 R22: - 2834.593301140667 Fraction[1] + Q0_var[1] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: Q0_var[1] + Lost_0_aux_1 = 615.0163626698273
 R25: Q0_var[1] + Q_hat_adjusted_1 = 682.9019267962094
 Constr_Q1_plus_left_1: - Left_0[1] - Q1_var[1] + Q1_plus_left[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: Q1_plus_left[1] + Lost_0_aux_1
   = 67.88545513387912
 Profit_Constraint_1: - 600 Sold_0[1] + 600 Lost_0[1] - 600 Sold_1[1]
   + 400 Left_1[1] + 600 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R32: - 16.66572576014321 alphas[0] - alphas[1]
   - 60.56848076433241 alphas[2] - alphas[3] + f_var[2] = 0
 R33: - 2834.593301140667 Fraction[2] + Q0_var[2] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: Q0_var[2] + Lost_0_aux_2 = 546.6986433271759
 R36: Q0_var[2] + Q_hat_adjusted_2 = 607.2672021267082
 Constr_Q1_plus_left_2: - Left_0[2] - Q1_var[2] + Q1_plus_left[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: Q1_plus_left[2] + Lost_0_aux_2
   = 60.56848067789765
 Profit_Constraint_2: - 600 Sold_0[2] + 600 Lost_0[2] - 600 Sold_1[2]
   + 400 Left_1[2] + 600 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R43: - 15.40550694993998 alphas[0] - alphas[1]
   - 66.96997241724988 alphas[2] - alphas[3] + f_var[3] = 0
 R44: - 2834.593301140667 Fraction[3] + Q0_var[3] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: Q0_var[3] + Lost_0_aux_3 = 609.0032336399942
 R47: Q0_var[3] + Q_hat_adjusted_3 = 675.9739478632746
 Constr_Q1_plus_left_3: - Left_0[3] - Q1_var[3] + Q1_plus_left[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: Q1_plus_left[3] + Lost_0_aux_3
   = 66.96997241344539
 Profit_Constraint_3: - 600 Sold_0[3] + 600 Lost_0[3] - 600 Sold_1[3]
   + 400 Left_1[3] + 600 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R54: - 17.03620585706904 alphas[0] - alphas[1]
   - 67.78695395941104 alphas[2] - alphas[3] + f_var[4] = 0
 R55: - 2834.593301140667 Fraction[4] + Q0_var[4] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: Q0_var[4] + Lost_0_aux_4 = 603.1259811892521
 R58: Q0_var[4] + Q_hat_adjusted_4 = 670.9133161320248
 Constr_Q1_plus_left_4: - Left_0[4] - Q1_var[4] + Q1_plus_left[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: Q1_plus_left[4] + Lost_0_aux_4
   = 67.78695396132478
 Profit_Constraint_4: - 600 Sold_0[4] + 600 Lost_0[4] - 600 Sold_1[4]
   + 400 Left_1[4] + 600 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R65: - 16.16117071085471 alphas[0] - alphas[1]
   - 67.77407561848753 alphas[2] - alphas[3] + f_var[5] = 0
 R66: - 2834.593301140667 Fraction[5] + Q0_var[5] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: Q0_var[5] + Lost_0_aux_5 = 606.0684163781129
 R69: Q0_var[5] + Q_hat_adjusted_5 = 673.8430435437551
 Constr_Q1_plus_left_5: - Left_0[5] - Q1_var[5] + Q1_plus_left[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: Q1_plus_left[5] + Lost_0_aux_5
   = 67.77407561838368
 Profit_Constraint_5: - 600 Sold_0[5] + 600 Lost_0[5] - 600 Sold_1[5]
   + 400 Left_1[5] + 600 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R76: - 15.66243817378992 alphas[0] - alphas[1]
   - 62.59422564345355 alphas[2] - alphas[3] + f_var[6] = 0
 R77: - 2834.593301140667 Fraction[6] + Q0_var[6] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: Q0_var[6] + Lost_0_aux_6 = 571.7073320724038
 R80: Q0_var[6] + Q_hat_adjusted_6 = 634.3014540064046
 Constr_Q1_plus_left_6: - Left_0[6] - Q1_var[6] + Q1_plus_left[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: Q1_plus_left[6] + Lost_0_aux_6
   = 62.59422564478501
 Profit_Constraint_6: - 600 Sold_0[6] + 600 Lost_0[6] - 600 Sold_1[6]
   + 400 Left_1[6] + 600 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R87: - 15.26713590893413 alphas[0] - alphas[1]
   - 63.73813137932562 alphas[2] - alphas[3] + f_var[7] = 0
 R88: - 2834.593301140667 Fraction[7] + Q0_var[7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: Q0_var[7] + Lost_0_aux_7 = 579.3694054600585
 R91: Q0_var[7] + Q_hat_adjusted_7 = 643.1074841435628
 Constr_Q1_plus_left_7: - Left_0[7] - Q1_var[7] + Q1_plus_left[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: Q1_plus_left[7] + Lost_0_aux_7
   = 63.7381313890866
 Profit_Constraint_7: - 600 Sold_0[7] + 600 Lost_0[7] - 600 Sold_1[7]
   + 400 Left_1[7] + 600 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R98: - 18.6279718210529 alphas[0] - alphas[1]
   - 65.87599635196389 alphas[2] - alphas[3] + f_var[8] = 0
 R99: - 2834.593301140667 Fraction[8] + Q0_var[8] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: Q0_var[8] + Lost_0_aux_8 = 586.0404239277433
 R102: Q0_var[8] + Q_hat_adjusted_8 = 651.916806868776
 Constr_Q1_plus_left_8: - Left_0[8] - Q1_var[8] + Q1_plus_left[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: Q1_plus_left[8] + Lost_0_aux_8
   = 65.87599635322354
 Profit_Constraint_8: - 600 Sold_0[8] + 600 Lost_0[8] - 600 Sold_1[8]
   + 400 Left_1[8] + 600 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R109: - 15.05713729312516 alphas[0] - alphas[1]
   - 62.72821902424467 alphas[2] - alphas[3] + f_var[9] = 0
 R110: - 2834.593301140667 Fraction[9] + Q0_var[9] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: Q0_var[9] + Lost_0_aux_9 = 563.3984160819591
 R113: Q0_var[9] + Q_hat_adjusted_9 = 626.1265236013052
 Constr_Q1_plus_left_9: - Left_0[9] - Q1_var[9] + Q1_plus_left[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: Q1_plus_left[9] + Lost_0_aux_9
   = 62.72821901342263
 Profit_Constraint_9: - 600 Sold_0[9] + 600 Lost_0[9] - 600 Sold_1[9]
   + 400 Left_1[9] + 600 Lost_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R120: - 18.85290374251388 alphas[0] - alphas[1]
   - 63.70852799217889 alphas[2] - alphas[3] + f_var[10] = 0
 R121: - 2834.593301140667 Fraction[10] + Q0_var[10] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: Q0_var[10] + Lost_0_aux_10 = 579.3341529840138
 R124: Q0_var[10] + Q_hat_adjusted_10 = 643.0423325948459
 Constr_Q1_plus_left_10: - Left_0[10] - Q1_var[10] + Q1_plus_left[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: Q1_plus_left[10] + Lost_0_aux_10
   = 63.70852813767267
 Profit_Constraint_10: - 600 Sold_0[10] + 600 Lost_0[10] - 600 Sold_1[10]
   + 400 Left_1[10] + 600 Lost_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R131: - 15.73473322700188 alphas[0] - alphas[1]
   - 61.9705428018564 alphas[2] - alphas[3] + f_var[11] = 0
 R132: - 2834.593301140667 Fraction[11] + Q0_var[11] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: Q0_var[11] + Lost_0_aux_11 = 563.2404568964613
 R135: Q0_var[11] + Q_hat_adjusted_11 = 625.2103841179132
 Constr_Q1_plus_left_11: - Left_0[11] - Q1_var[11] + Q1_plus_left[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: Q1_plus_left[11] + Lost_0_aux_11
   = 61.97054286979089
 Profit_Constraint_11: - 600 Sold_0[11] + 600 Lost_0[11] - 600 Sold_1[11]
   + 400 Left_1[11] + 600 Lost_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R142: - 15.39761041293378 alphas[0] - alphas[1]
   - 64.59855883756008 alphas[2] - alphas[3] + f_var[12] = 0
 R143: - 2834.593301140667 Fraction[12] + Q0_var[12] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: Q0_var[12] + Lost_0_aux_12 = 589.8157338007982
 R146: Q0_var[12] + Q_hat_adjusted_12 = 654.4143044644692
 Constr_Q1_plus_left_12: - Left_0[12] - Q1_var[12] + Q1_plus_left[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: Q1_plus_left[12] + Lost_0_aux_12
   = 64.59855883545265
 Profit_Constraint_12: - 600 Sold_0[12] + 600 Lost_0[12] - 600 Sold_1[12]
   + 400 Left_1[12] + 600 Lost_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R153: - 15.44801517119303 alphas[0] - alphas[1]
   - 60.44612301254114 alphas[2] - alphas[3] + f_var[13] = 0
 R154: - 2834.593301140667 Fraction[13] + Q0_var[13] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: Q0_var[13] + Lost_0_aux_13 = 542.5399986686749
 R157: Q0_var[13] + Q_hat_adjusted_13 = 602.9849285304032
 Constr_Q1_plus_left_13: - Left_0[13] - Q1_var[13] + Q1_plus_left[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: Q1_plus_left[13] + Lost_0_aux_13
   = 60.44612296107023
 Profit_Constraint_13: - 600 Sold_0[13] + 600 Lost_0[13] - 600 Sold_1[13]
   + 400 Left_1[13] + 600 Lost_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R164: - 18.36023903676957 alphas[0] - alphas[1]
   - 67.99795884570618 alphas[2] - alphas[3] + f_var[14] = 0
 R165: - 2834.593301140667 Fraction[14] + Q0_var[14] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: Q0_var[14] + Lost_0_aux_14 = 607.6836070631404
 R168: Q0_var[14] + Q_hat_adjusted_14 = 675.6817537595296
 Constr_Q1_plus_left_14: - Left_0[14] - Q1_var[14] + Q1_plus_left[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: Q1_plus_left[14] + Lost_0_aux_14
   = 67.99795884257753
 Profit_Constraint_14: - 600 Sold_0[14] + 600 Lost_0[14] - 600 Sold_1[14]
   + 400 Left_1[14] + 600 Lost_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R175: - 16.22683604926423 alphas[0] - alphas[1]
   - 60.76956446986632 alphas[2] - alphas[3] + f_var[15] = 0
 R176: - 2834.593301140667 Fraction[15] + Q0_var[15] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: Q0_var[15] + Lost_0_aux_15 = 558.4999511092925
 R179: Q0_var[15] + Q_hat_adjusted_15 = 619.268530867233
 Constr_Q1_plus_left_15: - Left_0[15] - Q1_var[15] + Q1_plus_left[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: Q1_plus_left[15] + Lost_0_aux_15
   = 60.76956452004073
 Profit_Constraint_15: - 600 Sold_0[15] + 600 Lost_0[15] - 600 Sold_1[15]
   + 400 Left_1[15] + 600 Lost_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R186: - 17.10269733340049 alphas[0] - alphas[1]
   - 65.18835148831526 alphas[2] - alphas[3] + f_var[16] = 0
 R187: - 2834.593301140667 Fraction[16] + Q0_var[16] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: Q0_var[16] + Lost_0_aux_16 = 592.945890446457
 R190: Q0_var[16] + Q_hat_adjusted_16 = 658.1342780874892
 Constr_Q1_plus_left_16: - Left_0[16] - Q1_var[16] + Q1_plus_left[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: Q1_plus_left[16] + Lost_0_aux_16
   = 65.18835150766253
 Profit_Constraint_16: - 600 Sold_0[16] + 600 Lost_0[16] - 600 Sold_1[16]
   + 400 Left_1[16] + 600 Lost_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R197: - 17.78684395661958 alphas[0] - alphas[1]
   - 63.06810099545196 alphas[2] - alphas[3] + f_var[17] = 0
 R198: - 2834.593301140667 Fraction[17] + Q0_var[17] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: Q0_var[17] + Lost_0_aux_17 = 575.3525948324216
 R201: Q0_var[17] + Q_hat_adjusted_17 = 638.4204619100359
 Constr_Q1_plus_left_17: - Left_0[17] - Q1_var[17] + Q1_plus_left[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: Q1_plus_left[17] + Lost_0_aux_17
   = 63.06810098016592
 Profit_Constraint_17: - 600 Sold_0[17] + 600 Lost_0[17] - 600 Sold_1[17]
   + 400 Left_1[17] + 600 Lost_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R208: - 19.30275586914397 alphas[0] - alphas[1]
   - 65.77542948831376 alphas[2] - alphas[3] + f_var[18] = 0
 R209: - 2834.593301140667 Fraction[18] + Q0_var[18] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: Q0_var[18] + Lost_0_aux_18 = 591.7522405037531
 R212: Q0_var[18] + Q_hat_adjusted_18 = 657.5276731291721
 Constr_Q1_plus_left_18: - Left_0[18] - Q1_var[18] + Q1_plus_left[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: Q1_plus_left[18] + Lost_0_aux_18
   = 65.77542945164845
 Profit_Constraint_18: - 600 Sold_0[18] + 600 Lost_0[18] - 600 Sold_1[18]
   + 400 Left_1[18] + 600 Lost_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R219: - 18.63522131355664 alphas[0] - alphas[1]
   - 69.59433340833425 alphas[2] - alphas[3] + f_var[19] = 0
 R220: - 2834.593301140667 Fraction[19] + Q0_var[19] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: Q0_var[19] + Lost_0_aux_19 = 611.9128922186179
 R223: Q0_var[19] + Q_hat_adjusted_19 = 681.506592651858
 Constr_Q1_plus_left_19: - Left_0[19] - Q1_var[19] + Q1_plus_left[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: Q1_plus_left[19] + Lost_0_aux_19
   = 69.5943334031505
 Profit_Constraint_19: - 600 Sold_0[19] + 600 Lost_0[19] - 600 Sold_1[19]
   + 400 Left_1[19] + 600 Lost_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R230: - 16.35163952619357 alphas[0] - alphas[1]
   - 66.45570244456003 alphas[2] - alphas[3] + f_var[20] = 0
 R231: - 2834.593301140667 Fraction[20] + Q0_var[20] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: Q0_var[20] + Lost_0_aux_20 = 593.5731668279178
 R234: Q0_var[20] + Q_hat_adjusted_20 = 660.029290883855
 Constr_Q1_plus_left_20: - Left_0[20] - Q1_var[20] + Q1_plus_left[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: Q1_plus_left[20] + Lost_0_aux_20
   = 66.45570244142786
 Profit_Constraint_20: - 600 Sold_0[20] + 600 Lost_0[20] - 600 Sold_1[20]
   + 400 Left_1[20] + 600 Lost_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R241: - 15.65741399645564 alphas[0] - alphas[1]
   - 60.35362435755491 alphas[2] - alphas[3] + f_var[21] = 0
 R242: - 2834.593301140667 Fraction[21] + Q0_var[21] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: Q0_var[21] + Lost_0_aux_21 = 550.9390774203794
 R245: Q0_var[21] + Q_hat_adjusted_21 = 611.2921820110839
 Constr_Q1_plus_left_21: - Left_0[21] - Q1_var[21] + Q1_plus_left[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: Q1_plus_left[21] + Lost_0_aux_21
   = 60.35362440554404
 Profit_Constraint_21: - 600 Sold_0[21] + 600 Lost_0[21] - 600 Sold_1[21]
   + 400 Left_1[21] + 600 Lost_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R252: - 15.27687160210599 alphas[0] - alphas[1]
   - 64.30402439508062 alphas[2] - alphas[3] + f_var[22] = 0
 R253: - 2834.593301140667 Fraction[22] + Q0_var[22] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: Q0_var[22] + Lost_0_aux_22 = 571.9932727014805
 R256: Q0_var[22] + Q_hat_adjusted_22 = 636.2972418287451
 Constr_Q1_plus_left_22: - Left_0[22] - Q1_var[22] + Q1_plus_left[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: Q1_plus_left[22] + Lost_0_aux_22
   = 64.30402439415226
 Profit_Constraint_22: - 600 Sold_0[22] + 600 Lost_0[22] - 600 Sold_1[22]
   + 400 Left_1[22] + 600 Lost_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R263: - 16.50799317240471 alphas[0] - alphas[1]
   - 65.1001685231825 alphas[2] - alphas[3] + f_var[23] = 0
 R264: - 2834.593301140667 Fraction[23] + Q0_var[23] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: Q0_var[23] + Lost_0_aux_23 = 570.1687370665331
 R267: Q0_var[23] + Q_hat_adjusted_23 = 635.269015506121
 Constr_Q1_plus_left_23: - Left_0[23] - Q1_var[23] + Q1_plus_left[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: Q1_plus_left[23] + Lost_0_aux_23
   = 65.10016851146773
 Profit_Constraint_23: - 600 Sold_0[23] + 600 Lost_0[23] - 600 Sold_1[23]
   + 400 Left_1[23] + 600 Lost_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R274: - 16.31059074619839 alphas[0] - alphas[1]
   - 65.36177494703452 alphas[2] - alphas[3] + f_var[24] = 0
 R275: - 2834.593301140667 Fraction[24] + Q0_var[24] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: Q0_var[24] + Lost_0_aux_24 = 583.9865505504501
 R278: Q0_var[24] + Q_hat_adjusted_24 = 649.3484153040766
 Constr_Q1_plus_left_24: - Left_0[24] - Q1_var[24] + Q1_plus_left[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: Q1_plus_left[24] + Lost_0_aux_24
   = 65.36177492848766
 Profit_Constraint_24: - 600 Sold_0[24] + 600 Lost_0[24] - 600 Sold_1[24]
   + 400 Left_1[24] + 600 Lost_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R285: - 17.2807028340024 alphas[0] - alphas[1]
   - 66.81392510603838 alphas[2] - alphas[3] + f_var[25] = 0
 R286: - 2834.593301140667 Fraction[25] + Q0_var[25] = 0
 Constr_Left_0_diff_aux_25: Sold_0[25] - Q0_var[25] + Left_0_aux_25 = 0
 Constr_Lost_0_diff_aux_25: Q0_var[25] + Lost_0_aux_25 = 591.2247489484268
 R289: Q0_var[25] + Q_hat_adjusted_25 = 658.0389693071218
 Constr_Q1_plus_left_25: - Left_0[25] - Q1_var[25] + Q1_plus_left[25] = 0
 Constr_Left_1_diff_aux_25: Sold_1[25] - Q1_plus_left[25] + Left_0_aux_25
   = 0
 Constr_Lost_1_diff_aux_25: Q1_plus_left[25] + Lost_0_aux_25
   = 66.81392510352053
 Profit_Constraint_25: - 600 Sold_0[25] + 600 Lost_0[25] - 600 Sold_1[25]
   + 400 Left_1[25] + 600 Lost_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R296: - 18.4164066777384 alphas[0] - alphas[1]
   - 62.77596097731766 alphas[2] - alphas[3] + f_var[26] = 0
 R297: - 2834.593301140667 Fraction[26] + Q0_var[26] = 0
 Constr_Left_0_diff_aux_26: Sold_0[26] - Q0_var[26] + Left_0_aux_26 = 0
 Constr_Lost_0_diff_aux_26: Q0_var[26] + Lost_0_aux_26 = 573.1127586650329
 R300: Q0_var[26] + Q_hat_adjusted_26 = 635.8880644061801
 Constr_Q1_plus_left_26: - Left_0[26] - Q1_var[26] + Q1_plus_left[26] = 0
 Constr_Left_1_diff_aux_26: Sold_1[26] - Q1_plus_left[26] + Left_0_aux_26
   = 0
 Constr_Lost_1_diff_aux_26: Q1_plus_left[26] + Lost_0_aux_26
   = 62.77596111419737
 Profit_Constraint_26: - 600 Sold_0[26] + 600 Lost_0[26] - 600 Sold_1[26]
   + 400 Left_1[26] + 600 Lost_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R307: - 18.47812722819429 alphas[0] - alphas[1]
   - 61.2886056546632 alphas[2] - alphas[3] + f_var[27] = 0
 R308: - 2834.593301140667 Fraction[27] + Q0_var[27] = 0
 Constr_Left_0_diff_aux_27: Sold_0[27] - Q0_var[27] + Left_0_aux_27 = 0
 Constr_Lost_0_diff_aux_27: Q0_var[27] + Lost_0_aux_27 = 554.8033256237609
 R311: Q0_var[27] + Q_hat_adjusted_27 = 616.091362013152
 Constr_Q1_plus_left_27: - Left_0[27] - Q1_var[27] + Q1_plus_left[27] = 0
 Constr_Left_1_diff_aux_27: Sold_1[27] - Q1_plus_left[27] + Left_0_aux_27
   = 0
 Constr_Lost_1_diff_aux_27: Q1_plus_left[27] + Lost_0_aux_27
   = 61.28860570136555
 Profit_Constraint_27: - 600 Sold_0[27] + 600 Lost_0[27] - 600 Sold_1[27]
   + 400 Left_1[27] + 600 Lost_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R318: - 16.41759423291083 alphas[0] - alphas[1]
   - 63.92675676547094 alphas[2] - alphas[3] + f_var[28] = 0
 R319: - 2834.593301140667 Fraction[28] + Q0_var[28] = 0
 Constr_Left_0_diff_aux_28: Sold_0[28] - Q0_var[28] + Left_0_aux_28 = 0
 Constr_Lost_0_diff_aux_28: Q0_var[28] + Lost_0_aux_28 = 581.1188735907376
 R322: Q0_var[28] + Q_hat_adjusted_28 = 645.0455301389331
 Constr_Q1_plus_left_28: - Left_0[28] - Q1_var[28] + Q1_plus_left[28] = 0
 Constr_Left_1_diff_aux_28: Sold_1[28] - Q1_plus_left[28] + Left_0_aux_28
   = 0
 Constr_Lost_1_diff_aux_28: Q1_plus_left[28] + Lost_0_aux_28
   = 63.92675674760295
 Profit_Constraint_28: - 600 Sold_0[28] + 600 Lost_0[28] - 600 Sold_1[28]
   + 400 Left_1[28] + 600 Lost_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R329: - 16.8996347795006 alphas[0] - alphas[1]
   - 69.56405722795949 alphas[2] - alphas[3] + f_var[29] = 0
 R330: - 2834.593301140667 Fraction[29] + Q0_var[29] = 0
 Constr_Left_0_diff_aux_29: Sold_0[29] - Q0_var[29] + Left_0_aux_29 = 0
 Constr_Lost_0_diff_aux_29: Q0_var[29] + Lost_0_aux_29 = 634.9397088867507
 R333: Q0_var[29] + Q_hat_adjusted_29 = 704.5043899276826
 Constr_Q1_plus_left_29: - Left_0[29] - Q1_var[29] + Q1_plus_left[29] = 0
 Constr_Left_1_diff_aux_29: Sold_1[29] - Q1_plus_left[29] + Left_0_aux_29
   = 0
 Constr_Lost_1_diff_aux_29: Q1_plus_left[29] + Lost_0_aux_29
   = 69.56405722872603
 Profit_Constraint_29: - 600 Sold_0[29] + 600 Lost_0[29] - 600 Sold_1[29]
   + 400 Left_1[29] + 600 Lost_1[29] + profits_vars[29] = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
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
 Q0_var[0] <= 2835.593301140667
 Q0_var[1] <= 2835.593301140667
 Q0_var[2] <= 2835.593301140667
 Q0_var[3] <= 2835.593301140667
 Q0_var[4] <= 2835.593301140667
 Q0_var[5] <= 2835.593301140667
 Q0_var[6] <= 2835.593301140667
 Q0_var[7] <= 2835.593301140667
 Q0_var[8] <= 2835.593301140667
 Q0_var[9] <= 2835.593301140667
 Q0_var[10] <= 2835.593301140667
 Q0_var[11] <= 2835.593301140667
 Q0_var[12] <= 2835.593301140667
 Q0_var[13] <= 2835.593301140667
 Q0_var[14] <= 2835.593301140667
 Q0_var[15] <= 2835.593301140667
 Q0_var[16] <= 2835.593301140667
 Q0_var[17] <= 2835.593301140667
 Q0_var[18] <= 2835.593301140667
 Q0_var[19] <= 2835.593301140667
 Q0_var[20] <= 2835.593301140667
 Q0_var[21] <= 2835.593301140667
 Q0_var[22] <= 2835.593301140667
 Q0_var[23] <= 2835.593301140667
 Q0_var[24] <= 2835.593301140667
 Q0_var[25] <= 2835.593301140667
 Q0_var[26] <= 2835.593301140667
 Q0_var[27] <= 2835.593301140667
 Q0_var[28] <= 2835.593301140667
 Q0_var[29] <= 2835.593301140667
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
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Q_hat_adjusted_0 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Q_hat_adjusted_1 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Q_hat_adjusted_2 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Q_hat_adjusted_3 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Q_hat_adjusted_4 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Q_hat_adjusted_5 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Q_hat_adjusted_6 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Q_hat_adjusted_7 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Q_hat_adjusted_8 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Q_hat_adjusted_9 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Q_hat_adjusted_10 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Q_hat_adjusted_11 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Q_hat_adjusted_12 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Q_hat_adjusted_13 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Q_hat_adjusted_14 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Q_hat_adjusted_15 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Q_hat_adjusted_16 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Q_hat_adjusted_17 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Q_hat_adjusted_18 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Q_hat_adjusted_19 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Q_hat_adjusted_20 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Q_hat_adjusted_21 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Q_hat_adjusted_22 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Q_hat_adjusted_23 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Q_hat_adjusted_24 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Q_hat_adjusted_25 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Q_hat_adjusted_26 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Q_hat_adjusted_27 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Q_hat_adjusted_28 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Q_hat_adjusted_29 free
General Constraints
 GC0: ( FuncNonlinear=1 ) Fraction[0] = LOGISTIC ( f_var[0] )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Q0_var[0] , 554.5300708824453 )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Q1_plus_left[0] , 61.81150966750985 )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 GC8: ( FuncNonlinear=1 ) Fraction[1] = LOGISTIC ( f_var[1] )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Q0_var[1] , 615.0163626698273 )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Q1_plus_left[1] , 67.88545513387912 )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 GC16: ( FuncNonlinear=1 ) Fraction[2] = LOGISTIC ( f_var[2] )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Q0_var[2] , 546.6986433271759 )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Q1_plus_left[2] , 60.56848067789765 )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 GC24: ( FuncNonlinear=1 ) Fraction[3] = LOGISTIC ( f_var[3] )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Q0_var[3] , 609.0032336399942 )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Q1_plus_left[3] , 66.96997241344539 )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 GC32: ( FuncNonlinear=1 ) Fraction[4] = LOGISTIC ( f_var[4] )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Q0_var[4] , 603.1259811892521 )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Q1_plus_left[4] , 67.78695396132478 )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 GC40: ( FuncNonlinear=1 ) Fraction[5] = LOGISTIC ( f_var[5] )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Q0_var[5] , 606.0684163781129 )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Q1_plus_left[5] , 67.77407561838368 )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 GC48: ( FuncNonlinear=1 ) Fraction[6] = LOGISTIC ( f_var[6] )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Q0_var[6] , 571.7073320724038 )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Q1_plus_left[6] , 62.59422564478501 )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 GC56: ( FuncNonlinear=1 ) Fraction[7] = LOGISTIC ( f_var[7] )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Q0_var[7] , 579.3694054600585 )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Q1_plus_left[7] , 63.7381313890866 )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 GC64: ( FuncNonlinear=1 ) Fraction[8] = LOGISTIC ( f_var[8] )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Q0_var[8] , 586.0404239277433 )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Q1_plus_left[8] , 65.87599635322354 )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 GC72: ( FuncNonlinear=1 ) Fraction[9] = LOGISTIC ( f_var[9] )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Q0_var[9] , 563.3984160819591 )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Q1_plus_left[9] , 62.72821901342263 )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 GC80: ( FuncNonlinear=1 ) Fraction[10] = LOGISTIC ( f_var[10] )
 Constr_Sold_0_min_10: Sold_0[10] = MIN ( Q0_var[10] , 579.3341529840138 )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Q1_plus_left[10]
   , 63.70852813767267 )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 GC88: ( FuncNonlinear=1 ) Fraction[11] = LOGISTIC ( f_var[11] )
 Constr_Sold_0_min_11: Sold_0[11] = MIN ( Q0_var[11] , 563.2404568964613 )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Q1_plus_left[11]
   , 61.97054286979089 )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 GC96: ( FuncNonlinear=1 ) Fraction[12] = LOGISTIC ( f_var[12] )
 Constr_Sold_0_min_12: Sold_0[12] = MIN ( Q0_var[12] , 589.8157338007982 )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Q1_plus_left[12]
   , 64.59855883545265 )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 GC104: ( FuncNonlinear=1 ) Fraction[13] = LOGISTIC ( f_var[13] )
 Constr_Sold_0_min_13: Sold_0[13] = MIN ( Q0_var[13] , 542.5399986686749 )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Q1_plus_left[13]
   , 60.44612296107023 )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 GC112: ( FuncNonlinear=1 ) Fraction[14] = LOGISTIC ( f_var[14] )
 Constr_Sold_0_min_14: Sold_0[14] = MIN ( Q0_var[14] , 607.6836070631404 )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Q1_plus_left[14]
   , 67.99795884257753 )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
 GC120: ( FuncNonlinear=1 ) Fraction[15] = LOGISTIC ( f_var[15] )
 Constr_Sold_0_min_15: Sold_0[15] = MIN ( Q0_var[15] , 558.4999511092925 )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Q1_plus_left[15]
   , 60.76956452004073 )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_0_aux_15 , 0 )
 GC128: ( FuncNonlinear=1 ) Fraction[16] = LOGISTIC ( f_var[16] )
 Constr_Sold_0_min_16: Sold_0[16] = MIN ( Q0_var[16] , 592.945890446457 )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Q1_plus_left[16]
   , 65.18835150766253 )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_0_aux_16 , 0 )
 GC136: ( FuncNonlinear=1 ) Fraction[17] = LOGISTIC ( f_var[17] )
 Constr_Sold_0_min_17: Sold_0[17] = MIN ( Q0_var[17] , 575.3525948324216 )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Q1_plus_left[17]
   , 63.06810098016592 )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_0_aux_17 , 0 )
 GC144: ( FuncNonlinear=1 ) Fraction[18] = LOGISTIC ( f_var[18] )
 Constr_Sold_0_min_18: Sold_0[18] = MIN ( Q0_var[18] , 591.7522405037531 )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Q1_plus_left[18]
   , 65.77542945164845 )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_0_aux_18 , 0 )
 GC152: ( FuncNonlinear=1 ) Fraction[19] = LOGISTIC ( f_var[19] )
 Constr_Sold_0_min_19: Sold_0[19] = MIN ( Q0_var[19] , 611.9128922186179 )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Q1_plus_left[19]
   , 69.5943334031505 )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_0_aux_19 , 0 )
 GC160: ( FuncNonlinear=1 ) Fraction[20] = LOGISTIC ( f_var[20] )
 Constr_Sold_0_min_20: Sold_0[20] = MIN ( Q0_var[20] , 593.5731668279178 )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Q1_plus_left[20]
   , 66.45570244142786 )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_0_aux_20 , 0 )
 GC168: ( FuncNonlinear=1 ) Fraction[21] = LOGISTIC ( f_var[21] )
 Constr_Sold_0_min_21: Sold_0[21] = MIN ( Q0_var[21] , 550.9390774203794 )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Q1_plus_left[21]
   , 60.35362440554404 )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_0_aux_21 , 0 )
 GC176: ( FuncNonlinear=1 ) Fraction[22] = LOGISTIC ( f_var[22] )
 Constr_Sold_0_min_22: Sold_0[22] = MIN ( Q0_var[22] , 571.9932727014805 )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Q1_plus_left[22]
   , 64.30402439415226 )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_0_aux_22 , 0 )
 GC184: ( FuncNonlinear=1 ) Fraction[23] = LOGISTIC ( f_var[23] )
 Constr_Sold_0_min_23: Sold_0[23] = MIN ( Q0_var[23] , 570.1687370665331 )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Q1_plus_left[23]
   , 65.10016851146773 )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_0_aux_23 , 0 )
 GC192: ( FuncNonlinear=1 ) Fraction[24] = LOGISTIC ( f_var[24] )
 Constr_Sold_0_min_24: Sold_0[24] = MIN ( Q0_var[24] , 583.9865505504501 )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Q1_plus_left[24]
   , 65.36177492848766 )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_0_aux_24 , 0 )
 GC200: ( FuncNonlinear=1 ) Fraction[25] = LOGISTIC ( f_var[25] )
 Constr_Sold_0_min_25: Sold_0[25] = MIN ( Q0_var[25] , 591.2247489484268 )
 Constr_Left_0_max_25: Left_0[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_0_max_25: Lost_0[25] = MAX ( Lost_0_aux_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 Constr_Sold_1_min_25: Sold_1[25] = MIN ( Q1_plus_left[25]
   , 66.81392510352053 )
 Constr_Left_1_max_25: Left_1[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_1_max_25: Lost_1[25] = MAX ( Lost_0_aux_25 , 0 )
 GC208: ( FuncNonlinear=1 ) Fraction[26] = LOGISTIC ( f_var[26] )
 Constr_Sold_0_min_26: Sold_0[26] = MIN ( Q0_var[26] , 573.1127586650329 )
 Constr_Left_0_max_26: Left_0[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_0_max_26: Lost_0[26] = MAX ( Lost_0_aux_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 Constr_Sold_1_min_26: Sold_1[26] = MIN ( Q1_plus_left[26]
   , 62.77596111419737 )
 Constr_Left_1_max_26: Left_1[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_1_max_26: Lost_1[26] = MAX ( Lost_0_aux_26 , 0 )
 GC216: ( FuncNonlinear=1 ) Fraction[27] = LOGISTIC ( f_var[27] )
 Constr_Sold_0_min_27: Sold_0[27] = MIN ( Q0_var[27] , 554.8033256237609 )
 Constr_Left_0_max_27: Left_0[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_0_max_27: Lost_0[27] = MAX ( Lost_0_aux_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 Constr_Sold_1_min_27: Sold_1[27] = MIN ( Q1_plus_left[27]
   , 61.28860570136555 )
 Constr_Left_1_max_27: Left_1[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_1_max_27: Lost_1[27] = MAX ( Lost_0_aux_27 , 0 )
 GC224: ( FuncNonlinear=1 ) Fraction[28] = LOGISTIC ( f_var[28] )
 Constr_Sold_0_min_28: Sold_0[28] = MIN ( Q0_var[28] , 581.1188735907376 )
 Constr_Left_0_max_28: Left_0[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_0_max_28: Lost_0[28] = MAX ( Lost_0_aux_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 Constr_Sold_1_min_28: Sold_1[28] = MIN ( Q1_plus_left[28]
   , 63.92675674760295 )
 Constr_Left_1_max_28: Left_1[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_1_max_28: Lost_1[28] = MAX ( Lost_0_aux_28 , 0 )
 GC232: ( FuncNonlinear=1 ) Fraction[29] = LOGISTIC ( f_var[29] )
 Constr_Sold_0_min_29: Sold_0[29] = MIN ( Q0_var[29] , 634.9397088867507 )
 Constr_Left_0_max_29: Left_0[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_0_max_29: Lost_0[29] = MAX ( Lost_0_aux_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 Constr_Sold_1_min_29: Sold_1[29] = MIN ( Q1_plus_left[29]
   , 69.56405722872603 )
 Constr_Left_1_max_29: Left_1[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_1_max_29: Lost_1[29] = MAX ( Lost_0_aux_29 , 0 )
End
