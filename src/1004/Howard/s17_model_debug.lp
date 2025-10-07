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
   + profits_vars[29] + profits_vars[30] + profits_vars[31]
   + profits_vars[32] + profits_vars[33] + profits_vars[34]
   + profits_vars[35] + profits_vars[36] + profits_vars[37]
   + profits_vars[38] + profits_vars[39] + profits_vars[40]
   + profits_vars[41] + profits_vars[42] + profits_vars[43]
   + profits_vars[44] + profits_vars[45] + profits_vars[46]
   + profits_vars[47] + profits_vars[48] + profits_vars[49]
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
 R10: - 16.13272766147923 alphas[0] - 260.5790687553841 alphas[2]
   - alphas[3] + f_var[0] = 0
 R11: - 2769.422558776328 Fraction[0] + Q0_var[0] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: Q0_var[0] + Lost_0_aux_0 = 2353.826460953877
 R14: Q0_var[0] + Q_hat_adjusted_0 = 2614.405529951549
 Constr_Q1_plus_left_0: - Left_0[0] - Q1_var[0] + Q1_plus_left[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: Q1_plus_left[0] + Lost_0_aux_0
   = 260.5790687553841
 Profit_Constraint_0: - 600 Sold_0[0] + 600 Lost_0[0] - 600 Sold_1[0]
   + 400 Left_1[0] + 600 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R21: - 19.97187417358469 alphas[0] - 275.0405849123835 alphas[2]
   - alphas[3] + f_var[1] = 0
 R22: - 2769.422558776328 Fraction[1] + Q0_var[1] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: Q0_var[1] + Lost_0_aux_1 = 2470.599653833541
 R25: Q0_var[1] + Q_hat_adjusted_1 = 2745.640239041329
 Constr_Q1_plus_left_1: - Left_0[1] - Q1_var[1] + Q1_plus_left[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: Q1_plus_left[1] + Lost_0_aux_1
   = 275.0405849123835
 Profit_Constraint_1: - 600 Sold_0[1] + 600 Lost_0[1] - 600 Sold_1[1]
   + 400 Left_1[1] + 600 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R32: - 16.67088801384685 alphas[0] - 276.3048174200271 alphas[2]
   - alphas[3] + f_var[2] = 0
 R33: - 2769.422558776328 Fraction[2] + Q0_var[2] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: Q0_var[2] + Lost_0_aux_2 = 2485.765618643578
 R36: Q0_var[2] + Q_hat_adjusted_2 = 2762.070436284425
 Constr_Q1_plus_left_2: - Left_0[2] - Q1_var[2] + Q1_plus_left[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: Q1_plus_left[2] + Lost_0_aux_2
   = 276.3048174200271
 Profit_Constraint_2: - 600 Sold_0[2] + 600 Lost_0[2] - 600 Sold_1[2]
   + 400 Left_1[2] + 600 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R43: - 15.80152608512716 alphas[0] - 260.6641002019486 alphas[2]
   - alphas[3] + f_var[3] = 0
 R44: - 2769.422558776328 Fraction[3] + Q0_var[3] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: Q0_var[3] + Lost_0_aux_3 = 2334.171649072276
 R47: Q0_var[3] + Q_hat_adjusted_3 = 2594.835749575708
 Constr_Q1_plus_left_3: - Left_0[3] - Q1_var[3] + Q1_plus_left[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: Q1_plus_left[3] + Lost_0_aux_3
   = 260.6641002019486
 Profit_Constraint_3: - 600 Sold_0[3] + 600 Lost_0[3] - 600 Sold_1[3]
   + 400 Left_1[3] + 600 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R54: - 19.02915412902744 alphas[0] - 274.6130583161191 alphas[2]
   - alphas[3] + f_var[4] = 0
 R55: - 2769.422558776328 Fraction[4] + Q0_var[4] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: Q0_var[4] + Lost_0_aux_4 = 2477.678356612502
 R58: Q0_var[4] + Q_hat_adjusted_4 = 2752.291415067746
 Constr_Q1_plus_left_4: - Left_0[4] - Q1_var[4] + Q1_plus_left[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: Q1_plus_left[4] + Lost_0_aux_4
   = 274.6130583161191
 Profit_Constraint_4: - 600 Sold_0[4] + 600 Lost_0[4] - 600 Sold_1[4]
   + 400 Left_1[4] + 600 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R65: - 15.2253491369525 alphas[0] - 252.6409948799999 alphas[2]
   - alphas[3] + f_var[5] = 0
 R66: - 2769.422558776328 Fraction[5] + Q0_var[5] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: Q0_var[5] + Lost_0_aux_5 = 2273.007720241237
 R69: Q0_var[5] + Q_hat_adjusted_5 = 2525.64871539237
 Constr_Q1_plus_left_5: - Left_0[5] - Q1_var[5] + Q1_plus_left[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: Q1_plus_left[5] + Lost_0_aux_5
   = 252.6409948799999
 Profit_Constraint_5: - 600 Sold_0[5] + 600 Lost_0[5] - 600 Sold_1[5]
   + 400 Left_1[5] + 600 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R76: - 15.33204375734381 alphas[0] - 250.0036224819246 alphas[2]
   - alphas[3] + f_var[6] = 0
 R77: - 2769.422558776328 Fraction[6] + Q0_var[6] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: Q0_var[6] + Lost_0_aux_6 = 2250.170324230824
 R80: Q0_var[6] + Q_hat_adjusted_6 = 2500.17394696634
 Constr_Q1_plus_left_6: - Left_0[6] - Q1_var[6] + Q1_plus_left[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: Q1_plus_left[6] + Lost_0_aux_6
   = 250.0036224819246
 Profit_Constraint_6: - 600 Sold_0[6] + 600 Lost_0[6] - 600 Sold_1[6]
   + 400 Left_1[6] + 600 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R87: - 16.48759389727183 alphas[0] - 294.0160965832185 alphas[2]
   - alphas[3] + f_var[7] = 0
 R88: - 2769.422558776328 Fraction[7] + Q0_var[7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: Q0_var[7] + Lost_0_aux_7 = 2643.604005520732
 R91: Q0_var[7] + Q_hat_adjusted_7 = 2937.620102315769
 Constr_Q1_plus_left_7: - Left_0[7] - Q1_var[7] + Q1_plus_left[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: Q1_plus_left[7] + Lost_0_aux_7
   = 294.0160965832185
 Profit_Constraint_7: - 600 Sold_0[7] + 600 Lost_0[7] - 600 Sold_1[7]
   + 400 Left_1[7] + 600 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R98: - 15.23700487317698 alphas[0] - 270.7794351043076 alphas[2]
   - alphas[3] + f_var[8] = 0
 R99: - 2769.422558776328 Fraction[8] + Q0_var[8] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: Q0_var[8] + Lost_0_aux_8 = 2429.568377958915
 R102: Q0_var[8] + Q_hat_adjusted_8 = 2700.34781331848
 Constr_Q1_plus_left_8: - Left_0[8] - Q1_var[8] + Q1_plus_left[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: Q1_plus_left[8] + Lost_0_aux_8
   = 270.7794351043076
 Profit_Constraint_8: - 600 Sold_0[8] + 600 Lost_0[8] - 600 Sold_1[8]
   + 400 Left_1[8] + 600 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R109: - 19.97856341946153 alphas[0] - 295.3709899476678 alphas[2]
   - alphas[3] + f_var[9] = 0
 R110: - 2769.422558776328 Fraction[9] + Q0_var[9] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: Q0_var[9] + Lost_0_aux_9 = 2651.984378918178
 R113: Q0_var[9] + Q_hat_adjusted_9 = 2947.355369101005
 Constr_Q1_plus_left_9: - Left_0[9] - Q1_var[9] + Q1_plus_left[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: Q1_plus_left[9] + Lost_0_aux_9
   = 295.3709899476678
 Profit_Constraint_9: - 600 Sold_0[9] + 600 Lost_0[9] - 600 Sold_1[9]
   + 400 Left_1[9] + 600 Lost_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R120: - 18.45564443041066 alphas[0] - 274.815367986413 alphas[2]
   - alphas[3] + f_var[10] = 0
 R121: - 2769.422558776328 Fraction[10] + Q0_var[10] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: Q0_var[10] + Lost_0_aux_10 = 2467.228070174807
 R124: Q0_var[10] + Q_hat_adjusted_10 = 2742.043438455449
 Constr_Q1_plus_left_10: - Left_0[10] - Q1_var[10] + Q1_plus_left[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: Q1_plus_left[10] + Lost_0_aux_10
   = 274.815367986413
 Profit_Constraint_10: - 600 Sold_0[10] + 600 Lost_0[10] - 600 Sold_1[10]
   + 400 Left_1[10] + 600 Lost_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R131: - 17.59793220428678 alphas[0] - 271.5554147294083 alphas[2]
   - alphas[3] + f_var[11] = 0
 R132: - 2769.422558776328 Fraction[11] + Q0_var[11] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: Q0_var[11] + Lost_0_aux_11 = 2437.348839748115
 R135: Q0_var[11] + Q_hat_adjusted_11 = 2708.904254748957
 Constr_Q1_plus_left_11: - Left_0[11] - Q1_var[11] + Q1_plus_left[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: Q1_plus_left[11] + Lost_0_aux_11
   = 271.5554147294083
 Profit_Constraint_11: - 600 Sold_0[11] + 600 Lost_0[11] - 600 Sold_1[11]
   + 400 Left_1[11] + 600 Lost_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R142: - 19.02583194978132 alphas[0] - 264.7956070329304 alphas[2]
   - alphas[3] + f_var[12] = 0
 R143: - 2769.422558776328 Fraction[12] + Q0_var[12] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: Q0_var[12] + Lost_0_aux_12 = 2383.927915523295
 R146: Q0_var[12] + Q_hat_adjusted_12 = 2648.723522913968
 Constr_Q1_plus_left_12: - Left_0[12] - Q1_var[12] + Q1_plus_left[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: Q1_plus_left[12] + Lost_0_aux_12
   = 264.7956070329304
 Profit_Constraint_12: - 600 Sold_0[12] + 600 Lost_0[12] - 600 Sold_1[12]
   + 400 Left_1[12] + 600 Lost_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R153: - 18.31243400305967 alphas[0] - 254.0844468796519 alphas[2]
   - alphas[3] + f_var[13] = 0
 R154: - 2769.422558776328 Fraction[13] + Q0_var[13] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: Q0_var[13] + Lost_0_aux_13 = 2279.049584175229
 R157: Q0_var[13] + Q_hat_adjusted_13 = 2533.13403153543
 Constr_Q1_plus_left_13: - Left_0[13] - Q1_var[13] + Q1_plus_left[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: Q1_plus_left[13] + Lost_0_aux_13
   = 254.0844468796519
 Profit_Constraint_13: - 600 Sold_0[13] + 600 Lost_0[13] - 600 Sold_1[13]
   + 400 Left_1[13] + 600 Lost_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R164: - 15.86632476837947 alphas[0] - 265.4847324632261 alphas[2]
   - alphas[3] + f_var[14] = 0
 R165: - 2769.422558776328 Fraction[14] + Q0_var[14] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: Q0_var[14] + Lost_0_aux_14 = 2389.266465060569
 R168: Q0_var[14] + Q_hat_adjusted_14 = 2654.751197792261
 Constr_Q1_plus_left_14: - Left_0[14] - Q1_var[14] + Q1_plus_left[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: Q1_plus_left[14] + Lost_0_aux_14
   = 265.4847324632261
 Profit_Constraint_14: - 600 Sold_0[14] + 600 Lost_0[14] - 600 Sold_1[14]
   + 400 Left_1[14] + 600 Lost_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R175: - 16.09816899824445 alphas[0] - 280.4235132586921 alphas[2]
   - alphas[3] + f_var[15] = 0
 R176: - 2769.422558776328 Fraction[15] + Q0_var[15] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: Q0_var[15] + Lost_0_aux_15 = 2523.949678387769
 R179: Q0_var[15] + Q_hat_adjusted_15 = 2804.37319190903
 Constr_Q1_plus_left_15: - Left_0[15] - Q1_var[15] + Q1_plus_left[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: Q1_plus_left[15] + Lost_0_aux_15
   = 280.4235132586921
 Profit_Constraint_15: - 600 Sold_0[15] + 600 Lost_0[15] - 600 Sold_1[15]
   + 400 Left_1[15] + 600 Lost_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R186: - 19.33847142860646 alphas[0] - 288.1393173611494 alphas[2]
   - alphas[3] + f_var[16] = 0
 R187: - 2769.422558776328 Fraction[16] + Q0_var[16] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: Q0_var[16] + Lost_0_aux_16 = 2590.230068143439
 R190: Q0_var[16] + Q_hat_adjusted_16 = 2878.36938575204
 Constr_Q1_plus_left_16: - Left_0[16] - Q1_var[16] + Q1_plus_left[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: Q1_plus_left[16] + Lost_0_aux_16
   = 288.1393173611494
 Profit_Constraint_16: - 600 Sold_0[16] + 600 Lost_0[16] - 600 Sold_1[16]
   + 400 Left_1[16] + 600 Lost_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R197: - 16.68480663386911 alphas[0] - 277.0412758826922 alphas[2]
   - alphas[3] + f_var[17] = 0
 R198: - 2769.422558776328 Fraction[17] + Q0_var[17] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: Q0_var[17] + Lost_0_aux_17 = 2495.644796933137
 R201: Q0_var[17] + Q_hat_adjusted_17 = 2772.686073109855
 Constr_Q1_plus_left_17: - Left_0[17] - Q1_var[17] + Q1_plus_left[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: Q1_plus_left[17] + Lost_0_aux_17
   = 277.0412758826922
 Profit_Constraint_17: - 600 Sold_0[17] + 600 Lost_0[17] - 600 Sold_1[17]
   + 400 Left_1[17] + 600 Lost_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R208: - 17.23820171312439 alphas[0] - 287.9127921848209 alphas[2]
   - alphas[3] + f_var[18] = 0
 R209: - 2769.422558776328 Fraction[18] + Q0_var[18] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: Q0_var[18] + Lost_0_aux_18 = 2592.915198591022
 R212: Q0_var[18] + Q_hat_adjusted_18 = 2880.827991027036
 Constr_Q1_plus_left_18: - Left_0[18] - Q1_var[18] + Q1_plus_left[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: Q1_plus_left[18] + Lost_0_aux_18
   = 287.9127921848209
 Profit_Constraint_18: - 600 Sold_0[18] + 600 Lost_0[18] - 600 Sold_1[18]
   + 400 Left_1[18] + 600 Lost_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R219: - 16.95567856061673 alphas[0] - 257.8875795829494 alphas[2]
   - alphas[3] + f_var[19] = 0
 R220: - 2769.422558776328 Fraction[19] + Q0_var[19] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: Q0_var[19] + Lost_0_aux_19 = 2317.979740718746
 R223: Q0_var[19] + Q_hat_adjusted_19 = 2575.867320503582
 Constr_Q1_plus_left_19: - Left_0[19] - Q1_var[19] + Q1_plus_left[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: Q1_plus_left[19] + Lost_0_aux_19
   = 257.8875795829494
 Profit_Constraint_19: - 600 Sold_0[19] + 600 Lost_0[19] - 600 Sold_1[19]
   + 400 Left_1[19] + 600 Lost_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R230: - 15.26014906002521 alphas[0] - 279.1619965839587 alphas[2]
   - alphas[3] + f_var[20] = 0
 R231: - 2769.422558776328 Fraction[20] + Q0_var[20] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: Q0_var[20] + Lost_0_aux_20 = 2512.028407170466
 R234: Q0_var[20] + Q_hat_adjusted_20 = 2791.190403965011
 Constr_Q1_plus_left_20: - Left_0[20] - Q1_var[20] + Q1_plus_left[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: Q1_plus_left[20] + Lost_0_aux_20
   = 279.1619965839587
 Profit_Constraint_20: - 600 Sold_0[20] + 600 Lost_0[20] - 600 Sold_1[20]
   + 400 Left_1[20] + 600 Lost_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R241: - 18.5228685864181 alphas[0] - 294.1301613512855 alphas[2]
   - alphas[3] + f_var[21] = 0
 R242: - 2769.422558776328 Fraction[21] + Q0_var[21] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: Q0_var[21] + Lost_0_aux_21 = 2654.219523781162
 R245: Q0_var[21] + Q_hat_adjusted_21 = 2948.349685366147
 Constr_Q1_plus_left_21: - Left_0[21] - Q1_var[21] + Q1_plus_left[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: Q1_plus_left[21] + Lost_0_aux_21
   = 294.1301613512855
 Profit_Constraint_21: - 600 Sold_0[21] + 600 Lost_0[21] - 600 Sold_1[21]
   + 400 Left_1[21] + 600 Lost_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R252: - 17.70037040393108 alphas[0] - 260.8980103820402 alphas[2]
   - alphas[3] + f_var[22] = 0
 R253: - 2769.422558776328 Fraction[22] + Q0_var[22] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: Q0_var[22] + Lost_0_aux_22 = 2344.494994709187
 R256: Q0_var[22] + Q_hat_adjusted_22 = 2605.393005342859
 Constr_Q1_plus_left_22: - Left_0[22] - Q1_var[22] + Q1_plus_left[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: Q1_plus_left[22] + Lost_0_aux_22
   = 260.8980103820402
 Profit_Constraint_22: - 600 Sold_0[22] + 600 Lost_0[22] - 600 Sold_1[22]
   + 400 Left_1[22] + 600 Lost_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R263: - 18.38070040906965 alphas[0] - 261.3834469196495 alphas[2]
   - alphas[3] + f_var[23] = 0
 R264: - 2769.422558776328 Fraction[23] + Q0_var[23] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: Q0_var[23] + Lost_0_aux_23 = 2342.394210356654
 R267: Q0_var[23] + Q_hat_adjusted_23 = 2603.777657386832
 Constr_Q1_plus_left_23: - Left_0[23] - Q1_var[23] + Q1_plus_left[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: Q1_plus_left[23] + Lost_0_aux_23
   = 261.3834469196495
 Profit_Constraint_23: - 600 Sold_0[23] + 600 Lost_0[23] - 600 Sold_1[23]
   + 400 Left_1[23] + 600 Lost_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R274: - 16.39115694833572 alphas[0] - 287.4216256534025 alphas[2]
   - alphas[3] + f_var[24] = 0
 R275: - 2769.422558776328 Fraction[24] + Q0_var[24] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: Q0_var[24] + Lost_0_aux_24 = 2589.209116826134
 R278: Q0_var[24] + Q_hat_adjusted_24 = 2876.630742769498
 Constr_Q1_plus_left_24: - Left_0[24] - Q1_var[24] + Q1_plus_left[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: Q1_plus_left[24] + Lost_0_aux_24
   = 287.4216256534025
 Profit_Constraint_24: - 600 Sold_0[24] + 600 Lost_0[24] - 600 Sold_1[24]
   + 400 Left_1[24] + 600 Lost_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R285: - 17.03456954163488 alphas[0] - 290.6028392424398 alphas[2]
   - alphas[3] + f_var[25] = 0
 R286: - 2769.422558776328 Fraction[25] + Q0_var[25] = 0
 Constr_Left_0_diff_aux_25: Sold_0[25] - Q0_var[25] + Left_0_aux_25 = 0
 Constr_Lost_0_diff_aux_25: Q0_var[25] + Lost_0_aux_25 = 2622.417767076949
 R289: Q0_var[25] + Q_hat_adjusted_25 = 2913.020606619118
 Constr_Q1_plus_left_25: - Left_0[25] - Q1_var[25] + Q1_plus_left[25] = 0
 Constr_Left_1_diff_aux_25: Sold_1[25] - Q1_plus_left[25] + Left_0_aux_25
   = 0
 Constr_Lost_1_diff_aux_25: Q1_plus_left[25] + Lost_0_aux_25
   = 290.6028392424398
 Profit_Constraint_25: - 600 Sold_0[25] + 600 Lost_0[25] - 600 Sold_1[25]
   + 400 Left_1[25] + 600 Lost_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R296: - 15.96324881135824 alphas[0] - 262.402316484705 alphas[2]
   - alphas[3] + f_var[26] = 0
 R297: - 2769.422558776328 Fraction[26] + Q0_var[26] = 0
 Constr_Left_0_diff_aux_26: Sold_0[26] - Q0_var[26] + Left_0_aux_26 = 0
 Constr_Lost_0_diff_aux_26: Q0_var[26] + Lost_0_aux_26 = 2364.239572032383
 R300: Q0_var[26] + Q_hat_adjusted_26 = 2626.641888834984
 Constr_Q1_plus_left_26: - Left_0[26] - Q1_var[26] + Q1_plus_left[26] = 0
 Constr_Left_1_diff_aux_26: Sold_1[26] - Q1_plus_left[26] + Left_0_aux_26
   = 0
 Constr_Lost_1_diff_aux_26: Q1_plus_left[26] + Lost_0_aux_26
   = 262.402316484705
 Profit_Constraint_26: - 600 Sold_0[26] + 600 Lost_0[26] - 600 Sold_1[26]
   + 400 Left_1[26] + 600 Lost_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R307: - 17.09841315718599 alphas[0] - 266.5784902093889 alphas[2]
   - alphas[3] + f_var[27] = 0
 R308: - 2769.422558776328 Fraction[27] + Q0_var[27] = 0
 Constr_Left_0_diff_aux_27: Sold_0[27] - Q0_var[27] + Left_0_aux_27 = 0
 Constr_Lost_0_diff_aux_27: Q0_var[27] + Lost_0_aux_27 = 2396.066424935766
 R311: Q0_var[27] + Q_hat_adjusted_27 = 2662.644915450199
 Constr_Q1_plus_left_27: - Left_0[27] - Q1_var[27] + Q1_plus_left[27] = 0
 Constr_Left_1_diff_aux_27: Sold_1[27] - Q1_plus_left[27] + Left_0_aux_27
   = 0
 Constr_Lost_1_diff_aux_27: Q1_plus_left[27] + Lost_0_aux_27
   = 266.5784902093889
 Profit_Constraint_27: - 600 Sold_0[27] + 600 Lost_0[27] - 600 Sold_1[27]
   + 400 Left_1[27] + 600 Lost_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R318: - 18.26745705898517 alphas[0] - 281.5884700225306 alphas[2]
   - alphas[3] + f_var[28] = 0
 R319: - 2769.422558776328 Fraction[28] + Q0_var[28] = 0
 Constr_Left_0_diff_aux_28: Sold_0[28] - Q0_var[28] + Left_0_aux_28 = 0
 Constr_Lost_0_diff_aux_28: Q0_var[28] + Lost_0_aux_28 = 2520.526273715069
 R322: Q0_var[28] + Q_hat_adjusted_28 = 2802.114743996124
 Constr_Q1_plus_left_28: - Left_0[28] - Q1_var[28] + Q1_plus_left[28] = 0
 Constr_Left_1_diff_aux_28: Sold_1[28] - Q1_plus_left[28] + Left_0_aux_28
   = 0
 Constr_Lost_1_diff_aux_28: Q1_plus_left[28] + Lost_0_aux_28
   = 281.5884700225306
 Profit_Constraint_28: - 600 Sold_0[28] + 600 Lost_0[28] - 600 Sold_1[28]
   + 400 Left_1[28] + 600 Lost_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R329: - 19.72344521596234 alphas[0] - 269.9215856594595 alphas[2]
   - alphas[3] + f_var[29] = 0
 R330: - 2769.422558776328 Fraction[29] + Q0_var[29] = 0
 Constr_Left_0_diff_aux_29: Sold_0[29] - Q0_var[29] + Left_0_aux_29 = 0
 Constr_Lost_0_diff_aux_29: Q0_var[29] + Lost_0_aux_29 = 2430.959127761711
 R333: Q0_var[29] + Q_hat_adjusted_29 = 2700.880713742849
 Constr_Q1_plus_left_29: - Left_0[29] - Q1_var[29] + Q1_plus_left[29] = 0
 Constr_Left_1_diff_aux_29: Sold_1[29] - Q1_plus_left[29] + Left_0_aux_29
   = 0
 Constr_Lost_1_diff_aux_29: Q1_plus_left[29] + Lost_0_aux_29
   = 269.9215856594595
 Profit_Constraint_29: - 600 Sold_0[29] + 600 Lost_0[29] - 600 Sold_1[29]
   + 400 Left_1[29] + 600 Lost_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R340: - 18.31728599943004 alphas[0] - 281.2005485064627 alphas[2]
   - alphas[3] + f_var[30] = 0
 R341: - 2769.422558776328 Fraction[30] + Q0_var[30] = 0
 Constr_Left_0_diff_aux_30: Sold_0[30] - Q0_var[30] + Left_0_aux_30 = 0
 Constr_Lost_0_diff_aux_30: Q0_var[30] + Lost_0_aux_30 = 2529.63806214751
 R344: Q0_var[30] + Q_hat_adjusted_30 = 2810.838610878257
 Constr_Q1_plus_left_30: - Left_0[30] - Q1_var[30] + Q1_plus_left[30] = 0
 Constr_Left_1_diff_aux_30: Sold_1[30] - Q1_plus_left[30] + Left_0_aux_30
   = 0
 Constr_Lost_1_diff_aux_30: Q1_plus_left[30] + Lost_0_aux_30
   = 281.2005485064627
 Profit_Constraint_30: - 600 Sold_0[30] + 600 Lost_0[30] - 600 Sold_1[30]
   + 400 Left_1[30] + 600 Lost_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R351: - 17.625798649965 alphas[0] - 274.8215016048363 alphas[2]
   - alphas[3] + f_var[31] = 0
 R352: - 2769.422558776328 Fraction[31] + Q0_var[31] = 0
 Constr_Left_0_diff_aux_31: Sold_0[31] - Q0_var[31] + Left_0_aux_31 = 0
 Constr_Lost_0_diff_aux_31: Q0_var[31] + Lost_0_aux_31 = 2469.247843816801
 R355: Q0_var[31] + Q_hat_adjusted_31 = 2744.069345688491
 Constr_Q1_plus_left_31: - Left_0[31] - Q1_var[31] + Q1_plus_left[31] = 0
 Constr_Left_1_diff_aux_31: Sold_1[31] - Q1_plus_left[31] + Left_0_aux_31
   = 0
 Constr_Lost_1_diff_aux_31: Q1_plus_left[31] + Lost_0_aux_31
   = 274.8215016048363
 Profit_Constraint_31: - 600 Sold_0[31] + 600 Lost_0[31] - 600 Sold_1[31]
   + 400 Left_1[31] + 600 Lost_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R362: - 17.08183611249081 alphas[0] - 288.6825654539363 alphas[2]
   - alphas[3] + f_var[32] = 0
 R363: - 2769.422558776328 Fraction[32] + Q0_var[32] = 0
 Constr_Left_0_diff_aux_32: Sold_0[32] - Q0_var[32] + Left_0_aux_32 = 0
 Constr_Lost_0_diff_aux_32: Q0_var[32] + Lost_0_aux_32 = 2588.463050407812
 R366: Q0_var[32] + Q_hat_adjusted_32 = 2877.145616146195
 Constr_Q1_plus_left_32: - Left_0[32] - Q1_var[32] + Q1_plus_left[32] = 0
 Constr_Left_1_diff_aux_32: Sold_1[32] - Q1_plus_left[32] + Left_0_aux_32
   = 0
 Constr_Lost_1_diff_aux_32: Q1_plus_left[32] + Lost_0_aux_32
   = 288.6825654539363
 Profit_Constraint_32: - 600 Sold_0[32] + 600 Lost_0[32] - 600 Sold_1[32]
   + 400 Left_1[32] + 600 Lost_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R373: - 15.5277484213049 alphas[0] - 292.2944323456662 alphas[2]
   - alphas[3] + f_var[33] = 0
 R374: - 2769.422558776328 Fraction[33] + Q0_var[33] = 0
 Constr_Left_0_diff_aux_33: Sold_0[33] - Q0_var[33] + Left_0_aux_33 = 0
 Constr_Lost_0_diff_aux_33: Q0_var[33] + Lost_0_aux_33 = 2636.059980144859
 R377: Q0_var[33] + Q_hat_adjusted_33 = 2928.35441270437
 Constr_Q1_plus_left_33: - Left_0[33] - Q1_var[33] + Q1_plus_left[33] = 0
 Constr_Left_1_diff_aux_33: Sold_1[33] - Q1_plus_left[33] + Left_0_aux_33
   = 0
 Constr_Lost_1_diff_aux_33: Q1_plus_left[33] + Lost_0_aux_33
   = 292.2944323456662
 Profit_Constraint_33: - 600 Sold_0[33] + 600 Lost_0[33] - 600 Sold_1[33]
   + 400 Left_1[33] + 600 Lost_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R384: - 17.38537148465995 alphas[0] - 293.191360465258 alphas[2]
   - alphas[3] + f_var[34] = 0
 R385: - 2769.422558776328 Fraction[34] + Q0_var[34] = 0
 Constr_Left_0_diff_aux_34: Sold_0[34] - Q0_var[34] + Left_0_aux_34 = 0
 Constr_Lost_0_diff_aux_34: Q0_var[34] + Lost_0_aux_34 = 2647.067690756923
 R388: Q0_var[34] + Q_hat_adjusted_34 = 2940.259051444384
 Constr_Q1_plus_left_34: - Left_0[34] - Q1_var[34] + Q1_plus_left[34] = 0
 Constr_Left_1_diff_aux_34: Sold_1[34] - Q1_plus_left[34] + Left_0_aux_34
   = 0
 Constr_Lost_1_diff_aux_34: Q1_plus_left[34] + Lost_0_aux_34
   = 293.191360465258
 Profit_Constraint_34: - 600 Sold_0[34] + 600 Lost_0[34] - 600 Sold_1[34]
   + 400 Left_1[34] + 600 Lost_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R395: - 15.77554046580322 alphas[0] - 280.5762538864922 alphas[2]
   - alphas[3] + f_var[35] = 0
 R396: - 2769.422558776328 Fraction[35] + Q0_var[35] = 0
 Constr_Left_0_diff_aux_35: Sold_0[35] - Q0_var[35] + Left_0_aux_35 = 0
 Constr_Lost_0_diff_aux_35: Q0_var[35] + Lost_0_aux_35 = 2527.238904841067
 R399: Q0_var[35] + Q_hat_adjusted_35 = 2807.815158936613
 Constr_Q1_plus_left_35: - Left_0[35] - Q1_var[35] + Q1_plus_left[35] = 0
 Constr_Left_1_diff_aux_35: Sold_1[35] - Q1_plus_left[35] + Left_0_aux_35
   = 0
 Constr_Lost_1_diff_aux_35: Q1_plus_left[35] + Lost_0_aux_35
   = 280.5762538864922
 Profit_Constraint_35: - 600 Sold_0[35] + 600 Lost_0[35] - 600 Sold_1[35]
   + 400 Left_1[35] + 600 Lost_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R406: - 17.91093591539545 alphas[0] - 280.6292052259625 alphas[2]
   - alphas[3] + f_var[36] = 0
 R407: - 2769.422558776328 Fraction[36] + Q0_var[36] = 0
 Constr_Left_0_diff_aux_36: Sold_0[36] - Q0_var[36] + Left_0_aux_36 = 0
 Constr_Lost_0_diff_aux_36: Q0_var[36] + Lost_0_aux_36 = 2525.291138551678
 R410: Q0_var[36] + Q_hat_adjusted_36 = 2805.920344063891
 Constr_Q1_plus_left_36: - Left_0[36] - Q1_var[36] + Q1_plus_left[36] = 0
 Constr_Left_1_diff_aux_36: Sold_1[36] - Q1_plus_left[36] + Left_0_aux_36
   = 0
 Constr_Lost_1_diff_aux_36: Q1_plus_left[36] + Lost_0_aux_36
   = 280.6292052259625
 Profit_Constraint_36: - 600 Sold_0[36] + 600 Lost_0[36] - 600 Sold_1[36]
   + 400 Left_1[36] + 600 Lost_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R417: - 18.65123538805292 alphas[0] - 272.4823449409703 alphas[2]
   - alphas[3] + f_var[37] = 0
 R418: - 2769.422558776328 Fraction[37] + Q0_var[37] = 0
 Constr_Left_0_diff_aux_37: Sold_0[37] - Q0_var[37] + Left_0_aux_37 = 0
 Constr_Lost_0_diff_aux_37: Q0_var[37] + Lost_0_aux_37 = 2461.701700308849
 R421: Q0_var[37] + Q_hat_adjusted_37 = 2734.184045539111
 Constr_Q1_plus_left_37: - Left_0[37] - Q1_var[37] + Q1_plus_left[37] = 0
 Constr_Left_1_diff_aux_37: Sold_1[37] - Q1_plus_left[37] + Left_0_aux_37
   = 0
 Constr_Lost_1_diff_aux_37: Q1_plus_left[37] + Lost_0_aux_37
   = 272.4823449409703
 Profit_Constraint_37: - 600 Sold_0[37] + 600 Lost_0[37] - 600 Sold_1[37]
   + 400 Left_1[37] + 600 Lost_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R428: - 19.73247805844096 alphas[0] - 251.3807022481653 alphas[2]
   - alphas[3] + f_var[38] = 0
 R429: - 2769.422558776328 Fraction[38] + Q0_var[38] = 0
 Constr_Left_0_diff_aux_38: Sold_0[38] - Q0_var[38] + Left_0_aux_38 = 0
 Constr_Lost_0_diff_aux_38: Q0_var[38] + Lost_0_aux_38 = 2260.18798818417
 R432: Q0_var[38] + Q_hat_adjusted_38 = 2511.568690291227
 Constr_Q1_plus_left_38: - Left_0[38] - Q1_var[38] + Q1_plus_left[38] = 0
 Constr_Left_1_diff_aux_38: Sold_1[38] - Q1_plus_left[38] + Left_0_aux_38
   = 0
 Constr_Lost_1_diff_aux_38: Q1_plus_left[38] + Lost_0_aux_38
   = 251.3807022481653
 Profit_Constraint_38: - 600 Sold_0[38] + 600 Lost_0[38] - 600 Sold_1[38]
   + 400 Left_1[38] + 600 Lost_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R439: - 16.48426433843983 alphas[0] - 264.9590109731631 alphas[2]
   - alphas[3] + f_var[39] = 0
 R440: - 2769.422558776328 Fraction[39] + Q0_var[39] = 0
 Constr_Left_0_diff_aux_39: Sold_0[39] - Q0_var[39] + Left_0_aux_39 = 0
 Constr_Lost_0_diff_aux_39: Q0_var[39] + Lost_0_aux_39 = 2387.82008807898
 R443: Q0_var[39] + Q_hat_adjusted_39 = 2652.779099324945
 Constr_Q1_plus_left_39: - Left_0[39] - Q1_var[39] + Q1_plus_left[39] = 0
 Constr_Left_1_diff_aux_39: Sold_1[39] - Q1_plus_left[39] + Left_0_aux_39
   = 0
 Constr_Lost_1_diff_aux_39: Q1_plus_left[39] + Lost_0_aux_39
   = 264.9590109731631
 Profit_Constraint_39: - 600 Sold_0[39] + 600 Lost_0[39] - 600 Sold_1[39]
   + 400 Left_1[39] + 600 Lost_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R450: - 16.98452769378767 alphas[0] - 253.184885925037 alphas[2]
   - alphas[3] + f_var[40] = 0
 R451: - 2769.422558776328 Fraction[40] + Q0_var[40] = 0
 Constr_Left_0_diff_aux_40: Sold_0[40] - Q0_var[40] + Left_0_aux_40 = 0
 Constr_Lost_0_diff_aux_40: Q0_var[40] + Lost_0_aux_40 = 2268.771684101584
 R454: Q0_var[40] + Q_hat_adjusted_40 = 2521.956570175915
 Constr_Q1_plus_left_40: - Left_0[40] - Q1_var[40] + Q1_plus_left[40] = 0
 Constr_Left_1_diff_aux_40: Sold_1[40] - Q1_plus_left[40] + Left_0_aux_40
   = 0
 Constr_Lost_1_diff_aux_40: Q1_plus_left[40] + Lost_0_aux_40
   = 253.184885925037
 Profit_Constraint_40: - 600 Sold_0[40] + 600 Lost_0[40] - 600 Sold_1[40]
   + 400 Left_1[40] + 600 Lost_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R461: - 15.91475969762666 alphas[0] - 271.8892624429877 alphas[2]
   - alphas[3] + f_var[41] = 0
 R462: - 2769.422558776328 Fraction[41] + Q0_var[41] = 0
 Constr_Left_0_diff_aux_41: Sold_0[41] - Q0_var[41] + Left_0_aux_41 = 0
 Constr_Lost_0_diff_aux_41: Q0_var[41] + Lost_0_aux_41 = 2444.918369613364
 R465: Q0_var[41] + Q_hat_adjusted_41 = 2716.807632363463
 Constr_Q1_plus_left_41: - Left_0[41] - Q1_var[41] + Q1_plus_left[41] = 0
 Constr_Left_1_diff_aux_41: Sold_1[41] - Q1_plus_left[41] + Left_0_aux_41
   = 0
 Constr_Lost_1_diff_aux_41: Q1_plus_left[41] + Lost_0_aux_41
   = 271.8892624429877
 Profit_Constraint_41: - 600 Sold_0[41] + 600 Lost_0[41] - 600 Sold_1[41]
   + 400 Left_1[41] + 600 Lost_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R472: - 19.38970254940216 alphas[0] - 292.4742624491307 alphas[2]
   - alphas[3] + f_var[42] = 0
 R473: - 2769.422558776328 Fraction[42] + Q0_var[42] = 0
 Constr_Left_0_diff_aux_42: Sold_0[42] - Q0_var[42] + Left_0_aux_42 = 0
 Constr_Lost_0_diff_aux_42: Q0_var[42] + Lost_0_aux_42 = 2629.396159804243
 R476: Q0_var[42] + Q_hat_adjusted_42 = 2921.870422447161
 Constr_Q1_plus_left_42: - Left_0[42] - Q1_var[42] + Q1_plus_left[42] = 0
 Constr_Left_1_diff_aux_42: Sold_1[42] - Q1_plus_left[42] + Left_0_aux_42
   = 0
 Constr_Lost_1_diff_aux_42: Q1_plus_left[42] + Lost_0_aux_42
   = 292.4742624491307
 Profit_Constraint_42: - 600 Sold_0[42] + 600 Lost_0[42] - 600 Sold_1[42]
   + 400 Left_1[42] + 600 Lost_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R483: - 19.12038612224712 alphas[0] - 279.840100885241 alphas[2]
   - alphas[3] + f_var[43] = 0
 R484: - 2769.422558776328 Fraction[43] + Q0_var[43] = 0
 Constr_Left_0_diff_aux_43: Sold_0[43] - Q0_var[43] + Left_0_aux_43 = 0
 Constr_Lost_0_diff_aux_43: Q0_var[43] + Lost_0_aux_43 = 2520.201464258795
 R487: Q0_var[43] + Q_hat_adjusted_43 = 2800.041565372443
 Constr_Q1_plus_left_43: - Left_0[43] - Q1_var[43] + Q1_plus_left[43] = 0
 Constr_Left_1_diff_aux_43: Sold_1[43] - Q1_plus_left[43] + Left_0_aux_43
   = 0
 Constr_Lost_1_diff_aux_43: Q1_plus_left[43] + Lost_0_aux_43
   = 279.840100885241
 Profit_Constraint_43: - 600 Sold_0[43] + 600 Lost_0[43] - 600 Sold_1[43]
   + 400 Left_1[43] + 600 Lost_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R494: - 18.87479947336618 alphas[0] - 283.7766299134778 alphas[2]
   - alphas[3] + f_var[44] = 0
 R495: - 2769.422558776328 Fraction[44] + Q0_var[44] = 0
 Constr_Left_0_diff_aux_44: Sold_0[44] - Q0_var[44] + Left_0_aux_44 = 0
 Constr_Lost_0_diff_aux_44: Q0_var[44] + Lost_0_aux_44 = 2554.992013735019
 R498: Q0_var[44] + Q_hat_adjusted_44 = 2838.768643859288
 Constr_Q1_plus_left_44: - Left_0[44] - Q1_var[44] + Q1_plus_left[44] = 0
 Constr_Left_1_diff_aux_44: Sold_1[44] - Q1_plus_left[44] + Left_0_aux_44
   = 0
 Constr_Lost_1_diff_aux_44: Q1_plus_left[44] + Lost_0_aux_44
   = 283.7766299134778
 Profit_Constraint_44: - 600 Sold_0[44] + 600 Lost_0[44] - 600 Sold_1[44]
   + 400 Left_1[44] + 600 Lost_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R505: - 15.28645019274951 alphas[0] - 259.4003985888538 alphas[2]
   - alphas[3] + f_var[45] = 0
 R506: - 2769.422558776328 Fraction[45] + Q0_var[45] = 0
 Constr_Left_0_diff_aux_45: Sold_0[45] - Q0_var[45] + Left_0_aux_45 = 0
 Constr_Lost_0_diff_aux_45: Q0_var[45] + Lost_0_aux_45 = 2326.931074156797
 R509: Q0_var[45] + Q_hat_adjusted_45 = 2586.331473024774
 Constr_Q1_plus_left_45: - Left_0[45] - Q1_var[45] + Q1_plus_left[45] = 0
 Constr_Left_1_diff_aux_45: Sold_1[45] - Q1_plus_left[45] + Left_0_aux_45
   = 0
 Constr_Lost_1_diff_aux_45: Q1_plus_left[45] + Lost_0_aux_45
   = 259.4003985888538
 Profit_Constraint_45: - 600 Sold_0[45] + 600 Lost_0[45] - 600 Sold_1[45]
   + 400 Left_1[45] + 600 Lost_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R516: - 17.78359720987461 alphas[0] - 263.0976574851397 alphas[2]
   - alphas[3] + f_var[46] = 0
 R517: - 2769.422558776328 Fraction[46] + Q0_var[46] = 0
 Constr_Left_0_diff_aux_46: Sold_0[46] - Q0_var[46] + Left_0_aux_46 = 0
 Constr_Lost_0_diff_aux_46: Q0_var[46] + Lost_0_aux_46 = 2350.496403114218
 R520: Q0_var[46] + Q_hat_adjusted_46 = 2613.594060872919
 Constr_Q1_plus_left_46: - Left_0[46] - Q1_var[46] + Q1_plus_left[46] = 0
 Constr_Left_1_diff_aux_46: Sold_1[46] - Q1_plus_left[46] + Left_0_aux_46
   = 0
 Constr_Lost_1_diff_aux_46: Q1_plus_left[46] + Lost_0_aux_46
   = 263.0976574851397
 Profit_Constraint_46: - 600 Sold_0[46] + 600 Lost_0[46] - 600 Sold_1[46]
   + 400 Left_1[46] + 600 Lost_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R527: - 16.17506781017735 alphas[0] - 258.1566076486448 alphas[2]
   - alphas[3] + f_var[47] = 0
 R528: - 2769.422558776328 Fraction[47] + Q0_var[47] = 0
 Constr_Left_0_diff_aux_47: Sold_0[47] - Q0_var[47] + Left_0_aux_47 = 0
 Constr_Lost_0_diff_aux_47: Q0_var[47] + Lost_0_aux_47 = 2321.51949020994
 R531: Q0_var[47] + Q_hat_adjusted_47 = 2579.676098203371
 Constr_Q1_plus_left_47: - Left_0[47] - Q1_var[47] + Q1_plus_left[47] = 0
 Constr_Left_1_diff_aux_47: Sold_1[47] - Q1_plus_left[47] + Left_0_aux_47
   = 0
 Constr_Lost_1_diff_aux_47: Q1_plus_left[47] + Lost_0_aux_47
   = 258.1566076486448
 Profit_Constraint_47: - 600 Sold_0[47] + 600 Lost_0[47] - 600 Sold_1[47]
   + 400 Left_1[47] + 600 Lost_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R538: - 15.59842598361153 alphas[0] - 292.1344381130215 alphas[2]
   - alphas[3] + f_var[48] = 0
 R539: - 2769.422558776328 Fraction[48] + Q0_var[48] = 0
 Constr_Left_0_diff_aux_48: Sold_0[48] - Q0_var[48] + Left_0_aux_48 = 0
 Constr_Lost_0_diff_aux_48: Q0_var[48] + Lost_0_aux_48 = 2629.057823384074
 R542: Q0_var[48] + Q_hat_adjusted_48 = 2921.192261780378
 Constr_Q1_plus_left_48: - Left_0[48] - Q1_var[48] + Q1_plus_left[48] = 0
 Constr_Left_1_diff_aux_48: Sold_1[48] - Q1_plus_left[48] + Left_0_aux_48
   = 0
 Constr_Lost_1_diff_aux_48: Q1_plus_left[48] + Lost_0_aux_48
   = 292.1344381130215
 Profit_Constraint_48: - 600 Sold_0[48] + 600 Lost_0[48] - 600 Sold_1[48]
   + 400 Left_1[48] + 600 Lost_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R549: - 18.90806550091695 alphas[0] - 264.0156506493746 alphas[2]
   - alphas[3] + f_var[49] = 0
 R550: - 2769.422558776328 Fraction[49] + Q0_var[49] = 0
 Constr_Left_0_diff_aux_49: Sold_0[49] - Q0_var[49] + Left_0_aux_49 = 0
 Constr_Lost_0_diff_aux_49: Q0_var[49] + Lost_0_aux_49 = 2387.266646296029
 R553: Q0_var[49] + Q_hat_adjusted_49 = 2651.282297259783
 Constr_Q1_plus_left_49: - Left_0[49] - Q1_var[49] + Q1_plus_left[49] = 0
 Constr_Left_1_diff_aux_49: Sold_1[49] - Q1_plus_left[49] + Left_0_aux_49
   = 0
 Constr_Lost_1_diff_aux_49: Q1_plus_left[49] + Lost_0_aux_49
   = 264.0156506493746
 Profit_Constraint_49: - 600 Sold_0[49] + 600 Lost_0[49] - 600 Sold_1[49]
   + 400 Left_1[49] + 600 Lost_1[49] + profits_vars[49] = 0
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
 Q0_var[0] <= 2770.422558776328
 Q0_var[1] <= 2770.422558776328
 Q0_var[2] <= 2770.422558776328
 Q0_var[3] <= 2770.422558776328
 Q0_var[4] <= 2770.422558776328
 Q0_var[5] <= 2770.422558776328
 Q0_var[6] <= 2770.422558776328
 Q0_var[7] <= 2770.422558776328
 Q0_var[8] <= 2770.422558776328
 Q0_var[9] <= 2770.422558776328
 Q0_var[10] <= 2770.422558776328
 Q0_var[11] <= 2770.422558776328
 Q0_var[12] <= 2770.422558776328
 Q0_var[13] <= 2770.422558776328
 Q0_var[14] <= 2770.422558776328
 Q0_var[15] <= 2770.422558776328
 Q0_var[16] <= 2770.422558776328
 Q0_var[17] <= 2770.422558776328
 Q0_var[18] <= 2770.422558776328
 Q0_var[19] <= 2770.422558776328
 Q0_var[20] <= 2770.422558776328
 Q0_var[21] <= 2770.422558776328
 Q0_var[22] <= 2770.422558776328
 Q0_var[23] <= 2770.422558776328
 Q0_var[24] <= 2770.422558776328
 Q0_var[25] <= 2770.422558776328
 Q0_var[26] <= 2770.422558776328
 Q0_var[27] <= 2770.422558776328
 Q0_var[28] <= 2770.422558776328
 Q0_var[29] <= 2770.422558776328
 Q0_var[30] <= 2770.422558776328
 Q0_var[31] <= 2770.422558776328
 Q0_var[32] <= 2770.422558776328
 Q0_var[33] <= 2770.422558776328
 Q0_var[34] <= 2770.422558776328
 Q0_var[35] <= 2770.422558776328
 Q0_var[36] <= 2770.422558776328
 Q0_var[37] <= 2770.422558776328
 Q0_var[38] <= 2770.422558776328
 Q0_var[39] <= 2770.422558776328
 Q0_var[40] <= 2770.422558776328
 Q0_var[41] <= 2770.422558776328
 Q0_var[42] <= 2770.422558776328
 Q0_var[43] <= 2770.422558776328
 Q0_var[44] <= 2770.422558776328
 Q0_var[45] <= 2770.422558776328
 Q0_var[46] <= 2770.422558776328
 Q0_var[47] <= 2770.422558776328
 Q0_var[48] <= 2770.422558776328
 Q0_var[49] <= 2770.422558776328
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
 Left_0_aux_30 free
 Lost_0_aux_30 free
 Left_0_aux_30 free
 Lost_0_aux_30 free
 Q_hat_adjusted_30 free
 Left_0_aux_31 free
 Lost_0_aux_31 free
 Left_0_aux_31 free
 Lost_0_aux_31 free
 Q_hat_adjusted_31 free
 Left_0_aux_32 free
 Lost_0_aux_32 free
 Left_0_aux_32 free
 Lost_0_aux_32 free
 Q_hat_adjusted_32 free
 Left_0_aux_33 free
 Lost_0_aux_33 free
 Left_0_aux_33 free
 Lost_0_aux_33 free
 Q_hat_adjusted_33 free
 Left_0_aux_34 free
 Lost_0_aux_34 free
 Left_0_aux_34 free
 Lost_0_aux_34 free
 Q_hat_adjusted_34 free
 Left_0_aux_35 free
 Lost_0_aux_35 free
 Left_0_aux_35 free
 Lost_0_aux_35 free
 Q_hat_adjusted_35 free
 Left_0_aux_36 free
 Lost_0_aux_36 free
 Left_0_aux_36 free
 Lost_0_aux_36 free
 Q_hat_adjusted_36 free
 Left_0_aux_37 free
 Lost_0_aux_37 free
 Left_0_aux_37 free
 Lost_0_aux_37 free
 Q_hat_adjusted_37 free
 Left_0_aux_38 free
 Lost_0_aux_38 free
 Left_0_aux_38 free
 Lost_0_aux_38 free
 Q_hat_adjusted_38 free
 Left_0_aux_39 free
 Lost_0_aux_39 free
 Left_0_aux_39 free
 Lost_0_aux_39 free
 Q_hat_adjusted_39 free
 Left_0_aux_40 free
 Lost_0_aux_40 free
 Left_0_aux_40 free
 Lost_0_aux_40 free
 Q_hat_adjusted_40 free
 Left_0_aux_41 free
 Lost_0_aux_41 free
 Left_0_aux_41 free
 Lost_0_aux_41 free
 Q_hat_adjusted_41 free
 Left_0_aux_42 free
 Lost_0_aux_42 free
 Left_0_aux_42 free
 Lost_0_aux_42 free
 Q_hat_adjusted_42 free
 Left_0_aux_43 free
 Lost_0_aux_43 free
 Left_0_aux_43 free
 Lost_0_aux_43 free
 Q_hat_adjusted_43 free
 Left_0_aux_44 free
 Lost_0_aux_44 free
 Left_0_aux_44 free
 Lost_0_aux_44 free
 Q_hat_adjusted_44 free
 Left_0_aux_45 free
 Lost_0_aux_45 free
 Left_0_aux_45 free
 Lost_0_aux_45 free
 Q_hat_adjusted_45 free
 Left_0_aux_46 free
 Lost_0_aux_46 free
 Left_0_aux_46 free
 Lost_0_aux_46 free
 Q_hat_adjusted_46 free
 Left_0_aux_47 free
 Lost_0_aux_47 free
 Left_0_aux_47 free
 Lost_0_aux_47 free
 Q_hat_adjusted_47 free
 Left_0_aux_48 free
 Lost_0_aux_48 free
 Left_0_aux_48 free
 Lost_0_aux_48 free
 Q_hat_adjusted_48 free
 Left_0_aux_49 free
 Lost_0_aux_49 free
 Left_0_aux_49 free
 Lost_0_aux_49 free
 Q_hat_adjusted_49 free
General Constraints
 GC0: ( FuncNonlinear=1 ) Fraction[0] = LOGISTIC ( f_var[0] )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Q0_var[0] , 2353.826460953877 )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Q1_plus_left[0] , 260.5790687553841 )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 GC8: ( FuncNonlinear=1 ) Fraction[1] = LOGISTIC ( f_var[1] )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Q0_var[1] , 2470.599653833541 )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Q1_plus_left[1] , 275.0405849123835 )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 GC16: ( FuncNonlinear=1 ) Fraction[2] = LOGISTIC ( f_var[2] )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Q0_var[2] , 2485.765618643578 )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Q1_plus_left[2] , 276.3048174200271 )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 GC24: ( FuncNonlinear=1 ) Fraction[3] = LOGISTIC ( f_var[3] )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Q0_var[3] , 2334.171649072276 )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Q1_plus_left[3] , 260.6641002019486 )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 GC32: ( FuncNonlinear=1 ) Fraction[4] = LOGISTIC ( f_var[4] )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Q0_var[4] , 2477.678356612502 )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Q1_plus_left[4] , 274.6130583161191 )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 GC40: ( FuncNonlinear=1 ) Fraction[5] = LOGISTIC ( f_var[5] )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Q0_var[5] , 2273.007720241237 )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Q1_plus_left[5] , 252.6409948799999 )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 GC48: ( FuncNonlinear=1 ) Fraction[6] = LOGISTIC ( f_var[6] )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Q0_var[6] , 2250.170324230824 )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Q1_plus_left[6] , 250.0036224819246 )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 GC56: ( FuncNonlinear=1 ) Fraction[7] = LOGISTIC ( f_var[7] )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Q0_var[7] , 2643.604005520732 )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Q1_plus_left[7] , 294.0160965832185 )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 GC64: ( FuncNonlinear=1 ) Fraction[8] = LOGISTIC ( f_var[8] )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Q0_var[8] , 2429.568377958915 )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Q1_plus_left[8] , 270.7794351043076 )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 GC72: ( FuncNonlinear=1 ) Fraction[9] = LOGISTIC ( f_var[9] )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Q0_var[9] , 2651.984378918178 )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Q1_plus_left[9] , 295.3709899476678 )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 GC80: ( FuncNonlinear=1 ) Fraction[10] = LOGISTIC ( f_var[10] )
 Constr_Sold_0_min_10: Sold_0[10] = MIN ( Q0_var[10] , 2467.228070174807 )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Q1_plus_left[10]
   , 274.815367986413 )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 GC88: ( FuncNonlinear=1 ) Fraction[11] = LOGISTIC ( f_var[11] )
 Constr_Sold_0_min_11: Sold_0[11] = MIN ( Q0_var[11] , 2437.348839748115 )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Q1_plus_left[11]
   , 271.5554147294083 )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 GC96: ( FuncNonlinear=1 ) Fraction[12] = LOGISTIC ( f_var[12] )
 Constr_Sold_0_min_12: Sold_0[12] = MIN ( Q0_var[12] , 2383.927915523295 )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Q1_plus_left[12]
   , 264.7956070329304 )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 GC104: ( FuncNonlinear=1 ) Fraction[13] = LOGISTIC ( f_var[13] )
 Constr_Sold_0_min_13: Sold_0[13] = MIN ( Q0_var[13] , 2279.049584175229 )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Q1_plus_left[13]
   , 254.0844468796519 )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 GC112: ( FuncNonlinear=1 ) Fraction[14] = LOGISTIC ( f_var[14] )
 Constr_Sold_0_min_14: Sold_0[14] = MIN ( Q0_var[14] , 2389.266465060569 )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Q1_plus_left[14]
   , 265.4847324632261 )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
 GC120: ( FuncNonlinear=1 ) Fraction[15] = LOGISTIC ( f_var[15] )
 Constr_Sold_0_min_15: Sold_0[15] = MIN ( Q0_var[15] , 2523.949678387769 )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Q1_plus_left[15]
   , 280.4235132586921 )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_0_aux_15 , 0 )
 GC128: ( FuncNonlinear=1 ) Fraction[16] = LOGISTIC ( f_var[16] )
 Constr_Sold_0_min_16: Sold_0[16] = MIN ( Q0_var[16] , 2590.230068143439 )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Q1_plus_left[16]
   , 288.1393173611494 )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_0_aux_16 , 0 )
 GC136: ( FuncNonlinear=1 ) Fraction[17] = LOGISTIC ( f_var[17] )
 Constr_Sold_0_min_17: Sold_0[17] = MIN ( Q0_var[17] , 2495.644796933137 )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Q1_plus_left[17]
   , 277.0412758826922 )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_0_aux_17 , 0 )
 GC144: ( FuncNonlinear=1 ) Fraction[18] = LOGISTIC ( f_var[18] )
 Constr_Sold_0_min_18: Sold_0[18] = MIN ( Q0_var[18] , 2592.915198591022 )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Q1_plus_left[18]
   , 287.9127921848209 )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_0_aux_18 , 0 )
 GC152: ( FuncNonlinear=1 ) Fraction[19] = LOGISTIC ( f_var[19] )
 Constr_Sold_0_min_19: Sold_0[19] = MIN ( Q0_var[19] , 2317.979740718746 )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Q1_plus_left[19]
   , 257.8875795829494 )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_0_aux_19 , 0 )
 GC160: ( FuncNonlinear=1 ) Fraction[20] = LOGISTIC ( f_var[20] )
 Constr_Sold_0_min_20: Sold_0[20] = MIN ( Q0_var[20] , 2512.028407170466 )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Q1_plus_left[20]
   , 279.1619965839587 )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_0_aux_20 , 0 )
 GC168: ( FuncNonlinear=1 ) Fraction[21] = LOGISTIC ( f_var[21] )
 Constr_Sold_0_min_21: Sold_0[21] = MIN ( Q0_var[21] , 2654.219523781162 )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Q1_plus_left[21]
   , 294.1301613512855 )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_0_aux_21 , 0 )
 GC176: ( FuncNonlinear=1 ) Fraction[22] = LOGISTIC ( f_var[22] )
 Constr_Sold_0_min_22: Sold_0[22] = MIN ( Q0_var[22] , 2344.494994709187 )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Q1_plus_left[22]
   , 260.8980103820402 )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_0_aux_22 , 0 )
 GC184: ( FuncNonlinear=1 ) Fraction[23] = LOGISTIC ( f_var[23] )
 Constr_Sold_0_min_23: Sold_0[23] = MIN ( Q0_var[23] , 2342.394210356654 )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Q1_plus_left[23]
   , 261.3834469196495 )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_0_aux_23 , 0 )
 GC192: ( FuncNonlinear=1 ) Fraction[24] = LOGISTIC ( f_var[24] )
 Constr_Sold_0_min_24: Sold_0[24] = MIN ( Q0_var[24] , 2589.209116826134 )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Q1_plus_left[24]
   , 287.4216256534025 )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_0_aux_24 , 0 )
 GC200: ( FuncNonlinear=1 ) Fraction[25] = LOGISTIC ( f_var[25] )
 Constr_Sold_0_min_25: Sold_0[25] = MIN ( Q0_var[25] , 2622.417767076949 )
 Constr_Left_0_max_25: Left_0[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_0_max_25: Lost_0[25] = MAX ( Lost_0_aux_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 Constr_Sold_1_min_25: Sold_1[25] = MIN ( Q1_plus_left[25]
   , 290.6028392424398 )
 Constr_Left_1_max_25: Left_1[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_1_max_25: Lost_1[25] = MAX ( Lost_0_aux_25 , 0 )
 GC208: ( FuncNonlinear=1 ) Fraction[26] = LOGISTIC ( f_var[26] )
 Constr_Sold_0_min_26: Sold_0[26] = MIN ( Q0_var[26] , 2364.239572032383 )
 Constr_Left_0_max_26: Left_0[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_0_max_26: Lost_0[26] = MAX ( Lost_0_aux_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 Constr_Sold_1_min_26: Sold_1[26] = MIN ( Q1_plus_left[26]
   , 262.402316484705 )
 Constr_Left_1_max_26: Left_1[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_1_max_26: Lost_1[26] = MAX ( Lost_0_aux_26 , 0 )
 GC216: ( FuncNonlinear=1 ) Fraction[27] = LOGISTIC ( f_var[27] )
 Constr_Sold_0_min_27: Sold_0[27] = MIN ( Q0_var[27] , 2396.066424935766 )
 Constr_Left_0_max_27: Left_0[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_0_max_27: Lost_0[27] = MAX ( Lost_0_aux_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 Constr_Sold_1_min_27: Sold_1[27] = MIN ( Q1_plus_left[27]
   , 266.5784902093889 )
 Constr_Left_1_max_27: Left_1[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_1_max_27: Lost_1[27] = MAX ( Lost_0_aux_27 , 0 )
 GC224: ( FuncNonlinear=1 ) Fraction[28] = LOGISTIC ( f_var[28] )
 Constr_Sold_0_min_28: Sold_0[28] = MIN ( Q0_var[28] , 2520.526273715069 )
 Constr_Left_0_max_28: Left_0[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_0_max_28: Lost_0[28] = MAX ( Lost_0_aux_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 Constr_Sold_1_min_28: Sold_1[28] = MIN ( Q1_plus_left[28]
   , 281.5884700225306 )
 Constr_Left_1_max_28: Left_1[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_1_max_28: Lost_1[28] = MAX ( Lost_0_aux_28 , 0 )
 GC232: ( FuncNonlinear=1 ) Fraction[29] = LOGISTIC ( f_var[29] )
 Constr_Sold_0_min_29: Sold_0[29] = MIN ( Q0_var[29] , 2430.959127761711 )
 Constr_Left_0_max_29: Left_0[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_0_max_29: Lost_0[29] = MAX ( Lost_0_aux_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 Constr_Sold_1_min_29: Sold_1[29] = MIN ( Q1_plus_left[29]
   , 269.9215856594595 )
 Constr_Left_1_max_29: Left_1[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_1_max_29: Lost_1[29] = MAX ( Lost_0_aux_29 , 0 )
 GC240: ( FuncNonlinear=1 ) Fraction[30] = LOGISTIC ( f_var[30] )
 Constr_Sold_0_min_30: Sold_0[30] = MIN ( Q0_var[30] , 2529.63806214751 )
 Constr_Left_0_max_30: Left_0[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_0_max_30: Lost_0[30] = MAX ( Lost_0_aux_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 Constr_Sold_1_min_30: Sold_1[30] = MIN ( Q1_plus_left[30]
   , 281.2005485064627 )
 Constr_Left_1_max_30: Left_1[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_1_max_30: Lost_1[30] = MAX ( Lost_0_aux_30 , 0 )
 GC248: ( FuncNonlinear=1 ) Fraction[31] = LOGISTIC ( f_var[31] )
 Constr_Sold_0_min_31: Sold_0[31] = MIN ( Q0_var[31] , 2469.247843816801 )
 Constr_Left_0_max_31: Left_0[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_0_max_31: Lost_0[31] = MAX ( Lost_0_aux_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 Constr_Sold_1_min_31: Sold_1[31] = MIN ( Q1_plus_left[31]
   , 274.8215016048363 )
 Constr_Left_1_max_31: Left_1[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_1_max_31: Lost_1[31] = MAX ( Lost_0_aux_31 , 0 )
 GC256: ( FuncNonlinear=1 ) Fraction[32] = LOGISTIC ( f_var[32] )
 Constr_Sold_0_min_32: Sold_0[32] = MIN ( Q0_var[32] , 2588.463050407812 )
 Constr_Left_0_max_32: Left_0[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_0_max_32: Lost_0[32] = MAX ( Lost_0_aux_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 Constr_Sold_1_min_32: Sold_1[32] = MIN ( Q1_plus_left[32]
   , 288.6825654539363 )
 Constr_Left_1_max_32: Left_1[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_1_max_32: Lost_1[32] = MAX ( Lost_0_aux_32 , 0 )
 GC264: ( FuncNonlinear=1 ) Fraction[33] = LOGISTIC ( f_var[33] )
 Constr_Sold_0_min_33: Sold_0[33] = MIN ( Q0_var[33] , 2636.059980144859 )
 Constr_Left_0_max_33: Left_0[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_0_max_33: Lost_0[33] = MAX ( Lost_0_aux_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 Constr_Sold_1_min_33: Sold_1[33] = MIN ( Q1_plus_left[33]
   , 292.2944323456662 )
 Constr_Left_1_max_33: Left_1[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_1_max_33: Lost_1[33] = MAX ( Lost_0_aux_33 , 0 )
 GC272: ( FuncNonlinear=1 ) Fraction[34] = LOGISTIC ( f_var[34] )
 Constr_Sold_0_min_34: Sold_0[34] = MIN ( Q0_var[34] , 2647.067690756923 )
 Constr_Left_0_max_34: Left_0[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_0_max_34: Lost_0[34] = MAX ( Lost_0_aux_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 Constr_Sold_1_min_34: Sold_1[34] = MIN ( Q1_plus_left[34]
   , 293.191360465258 )
 Constr_Left_1_max_34: Left_1[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_1_max_34: Lost_1[34] = MAX ( Lost_0_aux_34 , 0 )
 GC280: ( FuncNonlinear=1 ) Fraction[35] = LOGISTIC ( f_var[35] )
 Constr_Sold_0_min_35: Sold_0[35] = MIN ( Q0_var[35] , 2527.238904841067 )
 Constr_Left_0_max_35: Left_0[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_0_max_35: Lost_0[35] = MAX ( Lost_0_aux_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 Constr_Sold_1_min_35: Sold_1[35] = MIN ( Q1_plus_left[35]
   , 280.5762538864922 )
 Constr_Left_1_max_35: Left_1[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_1_max_35: Lost_1[35] = MAX ( Lost_0_aux_35 , 0 )
 GC288: ( FuncNonlinear=1 ) Fraction[36] = LOGISTIC ( f_var[36] )
 Constr_Sold_0_min_36: Sold_0[36] = MIN ( Q0_var[36] , 2525.291138551678 )
 Constr_Left_0_max_36: Left_0[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_0_max_36: Lost_0[36] = MAX ( Lost_0_aux_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 Constr_Sold_1_min_36: Sold_1[36] = MIN ( Q1_plus_left[36]
   , 280.6292052259625 )
 Constr_Left_1_max_36: Left_1[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_1_max_36: Lost_1[36] = MAX ( Lost_0_aux_36 , 0 )
 GC296: ( FuncNonlinear=1 ) Fraction[37] = LOGISTIC ( f_var[37] )
 Constr_Sold_0_min_37: Sold_0[37] = MIN ( Q0_var[37] , 2461.701700308849 )
 Constr_Left_0_max_37: Left_0[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_0_max_37: Lost_0[37] = MAX ( Lost_0_aux_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 Constr_Sold_1_min_37: Sold_1[37] = MIN ( Q1_plus_left[37]
   , 272.4823449409703 )
 Constr_Left_1_max_37: Left_1[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_1_max_37: Lost_1[37] = MAX ( Lost_0_aux_37 , 0 )
 GC304: ( FuncNonlinear=1 ) Fraction[38] = LOGISTIC ( f_var[38] )
 Constr_Sold_0_min_38: Sold_0[38] = MIN ( Q0_var[38] , 2260.18798818417 )
 Constr_Left_0_max_38: Left_0[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_0_max_38: Lost_0[38] = MAX ( Lost_0_aux_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 Constr_Sold_1_min_38: Sold_1[38] = MIN ( Q1_plus_left[38]
   , 251.3807022481653 )
 Constr_Left_1_max_38: Left_1[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_1_max_38: Lost_1[38] = MAX ( Lost_0_aux_38 , 0 )
 GC312: ( FuncNonlinear=1 ) Fraction[39] = LOGISTIC ( f_var[39] )
 Constr_Sold_0_min_39: Sold_0[39] = MIN ( Q0_var[39] , 2387.82008807898 )
 Constr_Left_0_max_39: Left_0[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_0_max_39: Lost_0[39] = MAX ( Lost_0_aux_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 Constr_Sold_1_min_39: Sold_1[39] = MIN ( Q1_plus_left[39]
   , 264.9590109731631 )
 Constr_Left_1_max_39: Left_1[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_1_max_39: Lost_1[39] = MAX ( Lost_0_aux_39 , 0 )
 GC320: ( FuncNonlinear=1 ) Fraction[40] = LOGISTIC ( f_var[40] )
 Constr_Sold_0_min_40: Sold_0[40] = MIN ( Q0_var[40] , 2268.771684101584 )
 Constr_Left_0_max_40: Left_0[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_0_max_40: Lost_0[40] = MAX ( Lost_0_aux_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 Constr_Sold_1_min_40: Sold_1[40] = MIN ( Q1_plus_left[40]
   , 253.184885925037 )
 Constr_Left_1_max_40: Left_1[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_1_max_40: Lost_1[40] = MAX ( Lost_0_aux_40 , 0 )
 GC328: ( FuncNonlinear=1 ) Fraction[41] = LOGISTIC ( f_var[41] )
 Constr_Sold_0_min_41: Sold_0[41] = MIN ( Q0_var[41] , 2444.918369613364 )
 Constr_Left_0_max_41: Left_0[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_0_max_41: Lost_0[41] = MAX ( Lost_0_aux_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 Constr_Sold_1_min_41: Sold_1[41] = MIN ( Q1_plus_left[41]
   , 271.8892624429877 )
 Constr_Left_1_max_41: Left_1[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_1_max_41: Lost_1[41] = MAX ( Lost_0_aux_41 , 0 )
 GC336: ( FuncNonlinear=1 ) Fraction[42] = LOGISTIC ( f_var[42] )
 Constr_Sold_0_min_42: Sold_0[42] = MIN ( Q0_var[42] , 2629.396159804243 )
 Constr_Left_0_max_42: Left_0[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_0_max_42: Lost_0[42] = MAX ( Lost_0_aux_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 Constr_Sold_1_min_42: Sold_1[42] = MIN ( Q1_plus_left[42]
   , 292.4742624491307 )
 Constr_Left_1_max_42: Left_1[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_1_max_42: Lost_1[42] = MAX ( Lost_0_aux_42 , 0 )
 GC344: ( FuncNonlinear=1 ) Fraction[43] = LOGISTIC ( f_var[43] )
 Constr_Sold_0_min_43: Sold_0[43] = MIN ( Q0_var[43] , 2520.201464258795 )
 Constr_Left_0_max_43: Left_0[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_0_max_43: Lost_0[43] = MAX ( Lost_0_aux_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 Constr_Sold_1_min_43: Sold_1[43] = MIN ( Q1_plus_left[43]
   , 279.840100885241 )
 Constr_Left_1_max_43: Left_1[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_1_max_43: Lost_1[43] = MAX ( Lost_0_aux_43 , 0 )
 GC352: ( FuncNonlinear=1 ) Fraction[44] = LOGISTIC ( f_var[44] )
 Constr_Sold_0_min_44: Sold_0[44] = MIN ( Q0_var[44] , 2554.992013735019 )
 Constr_Left_0_max_44: Left_0[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_0_max_44: Lost_0[44] = MAX ( Lost_0_aux_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 Constr_Sold_1_min_44: Sold_1[44] = MIN ( Q1_plus_left[44]
   , 283.7766299134778 )
 Constr_Left_1_max_44: Left_1[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_1_max_44: Lost_1[44] = MAX ( Lost_0_aux_44 , 0 )
 GC360: ( FuncNonlinear=1 ) Fraction[45] = LOGISTIC ( f_var[45] )
 Constr_Sold_0_min_45: Sold_0[45] = MIN ( Q0_var[45] , 2326.931074156797 )
 Constr_Left_0_max_45: Left_0[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_0_max_45: Lost_0[45] = MAX ( Lost_0_aux_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 Constr_Sold_1_min_45: Sold_1[45] = MIN ( Q1_plus_left[45]
   , 259.4003985888538 )
 Constr_Left_1_max_45: Left_1[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_1_max_45: Lost_1[45] = MAX ( Lost_0_aux_45 , 0 )
 GC368: ( FuncNonlinear=1 ) Fraction[46] = LOGISTIC ( f_var[46] )
 Constr_Sold_0_min_46: Sold_0[46] = MIN ( Q0_var[46] , 2350.496403114218 )
 Constr_Left_0_max_46: Left_0[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_0_max_46: Lost_0[46] = MAX ( Lost_0_aux_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 Constr_Sold_1_min_46: Sold_1[46] = MIN ( Q1_plus_left[46]
   , 263.0976574851397 )
 Constr_Left_1_max_46: Left_1[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_1_max_46: Lost_1[46] = MAX ( Lost_0_aux_46 , 0 )
 GC376: ( FuncNonlinear=1 ) Fraction[47] = LOGISTIC ( f_var[47] )
 Constr_Sold_0_min_47: Sold_0[47] = MIN ( Q0_var[47] , 2321.51949020994 )
 Constr_Left_0_max_47: Left_0[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_0_max_47: Lost_0[47] = MAX ( Lost_0_aux_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 Constr_Sold_1_min_47: Sold_1[47] = MIN ( Q1_plus_left[47]
   , 258.1566076486448 )
 Constr_Left_1_max_47: Left_1[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_1_max_47: Lost_1[47] = MAX ( Lost_0_aux_47 , 0 )
 GC384: ( FuncNonlinear=1 ) Fraction[48] = LOGISTIC ( f_var[48] )
 Constr_Sold_0_min_48: Sold_0[48] = MIN ( Q0_var[48] , 2629.057823384074 )
 Constr_Left_0_max_48: Left_0[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_0_max_48: Lost_0[48] = MAX ( Lost_0_aux_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 Constr_Sold_1_min_48: Sold_1[48] = MIN ( Q1_plus_left[48]
   , 292.1344381130215 )
 Constr_Left_1_max_48: Left_1[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_1_max_48: Lost_1[48] = MAX ( Lost_0_aux_48 , 0 )
 GC392: ( FuncNonlinear=1 ) Fraction[49] = LOGISTIC ( f_var[49] )
 Constr_Sold_0_min_49: Sold_0[49] = MIN ( Q0_var[49] , 2387.266646296029 )
 Constr_Left_0_max_49: Left_0[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_0_max_49: Lost_0[49] = MAX ( Lost_0_aux_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 Constr_Sold_1_min_49: Sold_1[49] = MIN ( Q1_plus_left[49]
   , 264.0156506493746 )
 Constr_Left_1_max_49: Left_1[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_1_max_49: Lost_1[49] = MAX ( Lost_0_aux_49 , 0 )
End
