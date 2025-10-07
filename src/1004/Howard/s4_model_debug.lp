\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 alphas[1] + 0 betas[0,1] + 0 betas[1,1] + 0 betas[2,1] + 0 betas[3,1]
   + 0 betas[4,1] + 0 betas[5,1] + 0 betas[6,1] + 0 betas[7,1]
   + 0 betas[8,1] + profits_vars[0] + profits_vars[1] + profits_vars[2]
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
 R0: - 15.19996400950359 alphas[0] - 294.4529976594864 alphas[2]
   - alphas[3] + f_var[0] = 0
 Q0_upper_bound_0: - 2806.409713277834 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_0_0: - 15.19996400950359 betas[0,0]
   - 294.4529976594864 betas[0,2] - betas[0,3] + tau[0,0] = 0
 tau_computation_0_1: - 15.19996400950359 betas[1,0]
   - 294.4529976594864 betas[1,2] - betas[1,3] + tau[0,1] = 0
 tau_computation_0_2: - 15.19996400950359 betas[2,0]
   - 294.4529976594864 betas[2,2] - betas[2,3] + tau[0,2] = 0
 tau_computation_0_3: - 15.19996400950359 betas[3,0]
   - 294.4529976594864 betas[3,2] - betas[3,3] + tau[0,3] = 0
 tau_computation_0_4: - 15.19996400950359 betas[4,0]
   - 294.4529976594864 betas[4,2] - betas[4,3] + tau[0,4] = 0
 tau_computation_0_5: - 15.19996400950359 betas[5,0]
   - 294.4529976594864 betas[5,2] - betas[5,3] + tau[0,5] = 0
 tau_computation_0_6: - 15.19996400950359 betas[6,0]
   - 294.4529976594864 betas[6,2] - betas[6,3] + tau[0,6] = 0
 tau_computation_0_7: - 15.19996400950359 betas[7,0]
   - 294.4529976594864 betas[7,2] - betas[7,3] + tau[0,7] = 0
 tau_computation_0_8: - 15.19996400950359 betas[8,0]
   - 294.4529976594864 betas[8,2] - betas[8,3] + tau[0,8] = 0
 tau_gap_0_0: tau[0,0] - 5e+06 R[0,0] - tau_star_0 <= -0.001
 tau_gap_0_1: tau[0,1] - 5e+06 R[0,1] - tau_star_0 <= -0.001
 tau_gap_0_2: tau[0,2] - 5e+06 R[0,2] - tau_star_0 <= -0.001
 tau_gap_0_3: tau[0,3] - 5e+06 R[0,3] - tau_star_0 <= -0.001
 tau_gap_0_4: tau[0,4] - 5e+06 R[0,4] - tau_star_0 <= -0.001
 tau_gap_0_5: tau[0,5] - 5e+06 R[0,5] - tau_star_0 <= -0.001
 tau_gap_0_6: tau[0,6] - 5e+06 R[0,6] - tau_star_0 <= -0.001
 tau_gap_0_7: tau[0,7] - 5e+06 R[0,7] - tau_star_0 <= -0.001
 tau_gap_0_8: tau[0,8] - 5e+06 R[0,8] - tau_star_0 <= -0.001
 one_R_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4] + R[0,5] + R[0,6]
   + R[0,7] + R[0,8] = 1
 Define_Q_hat_0: - 2927.646023594442 R[0,0] - 2941.735532999278 R[0,1]
   - 2941.865343660515 R[0,2] - 2941.865343444786 R[0,3]
   - 2945.595642563708 R[0,4] - 2945.59564251317 R[0,5]
   - 2947.129520369055 R[0,6] - 2942.752106790207 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 293.587276980768 R[0,0] - 588.0402746275458 R[0,1]
   - 880.9271745157705 R[0,2] - 1175.380172163812 R[0,3]
   - 1472.216323479793 R[0,4] - 1766.669321139279 R[0,5]
   - 2065.043415090014 R[0,6] - 2353.846110820609 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 2649.164829158814 R[0,0] - 2354.711831512036 R[0,1]
   - 2061.824931623812 R[0,2] - 1767.37193397577 R[0,3]
   - 1470.53578265979 R[0,4] - 1176.082785000303 R[0,5]
   - 877.7086910495681 R[0,6] - 588.9059953189728 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_1_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_1_aux_0 = 0
 Profit_Constraint_0: - 600 Sold_0[0] + 600 Lost_0[0] - 600 Sold_1[0]
   + 400 Left_1[0] + 600 Lost_1[0] + profits_vars[0] = 0
 R31: - 18.19852553037563 alphas[0] - 286.8639289857084 alphas[2]
   - alphas[3] + f_var[1] = 0
 Q0_upper_bound_1: - 2806.409713277834 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_1_0: - 18.19852553037563 betas[0,0]
   - 286.8639289857084 betas[0,2] - betas[0,3] + tau[1,0] = 0
 tau_computation_1_1: - 18.19852553037563 betas[1,0]
   - 286.8639289857084 betas[1,2] - betas[1,3] + tau[1,1] = 0
 tau_computation_1_2: - 18.19852553037563 betas[2,0]
   - 286.8639289857084 betas[2,2] - betas[2,3] + tau[1,2] = 0
 tau_computation_1_3: - 18.19852553037563 betas[3,0]
   - 286.8639289857084 betas[3,2] - betas[3,3] + tau[1,3] = 0
 tau_computation_1_4: - 18.19852553037563 betas[4,0]
   - 286.8639289857084 betas[4,2] - betas[4,3] + tau[1,4] = 0
 tau_computation_1_5: - 18.19852553037563 betas[5,0]
   - 286.8639289857084 betas[5,2] - betas[5,3] + tau[1,5] = 0
 tau_computation_1_6: - 18.19852553037563 betas[6,0]
   - 286.8639289857084 betas[6,2] - betas[6,3] + tau[1,6] = 0
 tau_computation_1_7: - 18.19852553037563 betas[7,0]
   - 286.8639289857084 betas[7,2] - betas[7,3] + tau[1,7] = 0
 tau_computation_1_8: - 18.19852553037563 betas[8,0]
   - 286.8639289857084 betas[8,2] - betas[8,3] + tau[1,8] = 0
 tau_gap_1_0: tau[1,0] - 5e+06 R[1,0] - tau_star_1 <= -0.001
 tau_gap_1_1: tau[1,1] - 5e+06 R[1,1] - tau_star_1 <= -0.001
 tau_gap_1_2: tau[1,2] - 5e+06 R[1,2] - tau_star_1 <= -0.001
 tau_gap_1_3: tau[1,3] - 5e+06 R[1,3] - tau_star_1 <= -0.001
 tau_gap_1_4: tau[1,4] - 5e+06 R[1,4] - tau_star_1 <= -0.001
 tau_gap_1_5: tau[1,5] - 5e+06 R[1,5] - tau_star_1 <= -0.001
 tau_gap_1_6: tau[1,6] - 5e+06 R[1,6] - tau_star_1 <= -0.001
 tau_gap_1_7: tau[1,7] - 5e+06 R[1,7] - tau_star_1 <= -0.001
 tau_gap_1_8: tau[1,8] - 5e+06 R[1,8] - tau_star_1 <= -0.001
 one_R_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4] + R[1,5] + R[1,6]
   + R[1,7] + R[1,8] = 1
 Define_Q_hat_1: - 2849.293216414115 R[1,0] - 2865.355928436874 R[1,1]
   - 2864.955283621469 R[1,2] - 2864.955283729544 R[1,3]
   - 2861.907022658345 R[1,4] - 2861.907022733612 R[1,5]
   - 2864.815447449215 R[1,6] - 2867.115222185494 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 285.1765892479286 R[1,0] - 572.0405182539309 R[1,1]
   - 861.1028348275529 R[1,2] - 1147.966763805785 R[1,3]
   - 1430.642912147452 R[1,4] - 1717.50684113316 R[1,5]
   - 2004.236023020991 R[1,6] - 2293.387363719255 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 2581.938632442743 R[1,0] - 2295.07470343674 R[1,1]
   - 2006.012386863119 R[1,2] - 1719.148457884886 R[1,3]
   - 1436.47230954322 R[1,4] - 1149.608380557512 R[1,5]
   - 862.8791986696802 R[1,6] - 573.7278579714168 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_1_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_1_aux_1 = 0
 Profit_Constraint_1: - 600 Sold_0[1] + 600 Lost_0[1] - 600 Sold_1[1]
   + 400 Left_1[1] + 600 Lost_1[1] + profits_vars[1] = 0
 R62: - 17.04151454169872 alphas[0] - 250.2576482134511 alphas[2]
   - alphas[3] + f_var[2] = 0
 Q0_upper_bound_2: - 2806.409713277834 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_2_0: - 17.04151454169872 betas[0,0]
   - 250.2576482134511 betas[0,2] - betas[0,3] + tau[2,0] = 0
 tau_computation_2_1: - 17.04151454169872 betas[1,0]
   - 250.2576482134511 betas[1,2] - betas[1,3] + tau[2,1] = 0
 tau_computation_2_2: - 17.04151454169872 betas[2,0]
   - 250.2576482134511 betas[2,2] - betas[2,3] + tau[2,2] = 0
 tau_computation_2_3: - 17.04151454169872 betas[3,0]
   - 250.2576482134511 betas[3,2] - betas[3,3] + tau[2,3] = 0
 tau_computation_2_4: - 17.04151454169872 betas[4,0]
   - 250.2576482134511 betas[4,2] - betas[4,3] + tau[2,4] = 0
 tau_computation_2_5: - 17.04151454169872 betas[5,0]
   - 250.2576482134511 betas[5,2] - betas[5,3] + tau[2,5] = 0
 tau_computation_2_6: - 17.04151454169872 betas[6,0]
   - 250.2576482134511 betas[6,2] - betas[6,3] + tau[2,6] = 0
 tau_computation_2_7: - 17.04151454169872 betas[7,0]
   - 250.2576482134511 betas[7,2] - betas[7,3] + tau[2,7] = 0
 tau_computation_2_8: - 17.04151454169872 betas[8,0]
   - 250.2576482134511 betas[8,2] - betas[8,3] + tau[2,8] = 0
 tau_gap_2_0: tau[2,0] - 5e+06 R[2,0] - tau_star_2 <= -0.001
 tau_gap_2_1: tau[2,1] - 5e+06 R[2,1] - tau_star_2 <= -0.001
 tau_gap_2_2: tau[2,2] - 5e+06 R[2,2] - tau_star_2 <= -0.001
 tau_gap_2_3: tau[2,3] - 5e+06 R[2,3] - tau_star_2 <= -0.001
 tau_gap_2_4: tau[2,4] - 5e+06 R[2,4] - tau_star_2 <= -0.001
 tau_gap_2_5: tau[2,5] - 5e+06 R[2,5] - tau_star_2 <= -0.001
 tau_gap_2_6: tau[2,6] - 5e+06 R[2,6] - tau_star_2 <= -0.001
 tau_gap_2_7: tau[2,7] - 5e+06 R[2,7] - tau_star_2 <= -0.001
 tau_gap_2_8: tau[2,8] - 5e+06 R[2,8] - tau_star_2 <= -0.001
 one_R_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4] + R[2,5] + R[2,6]
   + R[2,7] + R[2,8] = 1
 Define_Q_hat_2: - 2567.323316198635 R[2,0] - 2511.187254999828 R[2,1]
   - 2510.871409302957 R[2,2] - 2510.871410539527 R[2,3]
   - 2506.781434399261 R[2,4] - 2506.781434796615 R[2,5]
   - 2505.875425470907 R[2,6] - 2503.359438513232 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 254.9088698627862 R[2,0] - 505.1665179115381 R[2,1]
   - 759.9684007770006 R[2,2] - 1010.226048982102 R[2,3]
   - 1256.975321245565 R[2,4] - 1507.232969459016 R[2,5]
   - 1753.698360829951 R[2,6] - 2002.844141665951 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 2248.450568230067 R[2,0] - 1998.192920181315 R[2,1]
   - 1743.391037315852 R[2,2] - 1493.133389110751 R[2,3]
   - 1246.384116847288 R[2,4] - 996.1264686338372 R[2,5]
   - 749.6610772629022 R[2,6] - 500.5152964269023 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_1_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_1_aux_2 = 0
 Profit_Constraint_2: - 600 Sold_0[2] + 600 Lost_0[2] - 600 Sold_1[2]
   + 400 Left_1[2] + 600 Lost_1[2] + profits_vars[2] = 0
 R93: - 16.88703286294444 alphas[0] - 284.7078925684563 alphas[2]
   - alphas[3] + f_var[3] = 0
 Q0_upper_bound_3: - 2806.409713277834 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_3_0: - 16.88703286294444 betas[0,0]
   - 284.7078925684563 betas[0,2] - betas[0,3] + tau[3,0] = 0
 tau_computation_3_1: - 16.88703286294444 betas[1,0]
   - 284.7078925684563 betas[1,2] - betas[1,3] + tau[3,1] = 0
 tau_computation_3_2: - 16.88703286294444 betas[2,0]
   - 284.7078925684563 betas[2,2] - betas[2,3] + tau[3,2] = 0
 tau_computation_3_3: - 16.88703286294444 betas[3,0]
   - 284.7078925684563 betas[3,2] - betas[3,3] + tau[3,3] = 0
 tau_computation_3_4: - 16.88703286294444 betas[4,0]
   - 284.7078925684563 betas[4,2] - betas[4,3] + tau[3,4] = 0
 tau_computation_3_5: - 16.88703286294444 betas[5,0]
   - 284.7078925684563 betas[5,2] - betas[5,3] + tau[3,5] = 0
 tau_computation_3_6: - 16.88703286294444 betas[6,0]
   - 284.7078925684563 betas[6,2] - betas[6,3] + tau[3,6] = 0
 tau_computation_3_7: - 16.88703286294444 betas[7,0]
   - 284.7078925684563 betas[7,2] - betas[7,3] + tau[3,7] = 0
 tau_computation_3_8: - 16.88703286294444 betas[8,0]
   - 284.7078925684563 betas[8,2] - betas[8,3] + tau[3,8] = 0
 tau_gap_3_0: tau[3,0] - 5e+06 R[3,0] - tau_star_3 <= -0.001
 tau_gap_3_1: tau[3,1] - 5e+06 R[3,1] - tau_star_3 <= -0.001
 tau_gap_3_2: tau[3,2] - 5e+06 R[3,2] - tau_star_3 <= -0.001
 tau_gap_3_3: tau[3,3] - 5e+06 R[3,3] - tau_star_3 <= -0.001
 tau_gap_3_4: tau[3,4] - 5e+06 R[3,4] - tau_star_3 <= -0.001
 tau_gap_3_5: tau[3,5] - 5e+06 R[3,5] - tau_star_3 <= -0.001
 tau_gap_3_6: tau[3,6] - 5e+06 R[3,6] - tau_star_3 <= -0.001
 tau_gap_3_7: tau[3,7] - 5e+06 R[3,7] - tau_star_3 <= -0.001
 tau_gap_3_8: tau[3,8] - 5e+06 R[3,8] - tau_star_3 <= -0.001
 one_R_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4] + R[3,5] + R[3,6]
   + R[3,7] + R[3,8] = 1
 Define_Q_hat_3: - 2820.817827600115 R[3,0] - 2842.733587358809 R[3,1]
   - 2842.700048932991 R[3,2] - 2842.70004892383 R[3,3]
   - 2844.996967509373 R[3,4] - 2844.996967484903 R[3,5]
   - 2846.688995049542 R[3,6] - 2841.147300616467 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 282.1199329262917 R[3,0] - 566.8278254689835 R[3,1]
   - 850.5948992066016 R[3,2] - 1135.3027917806 R[3,3]
   - 1421.180408639315 R[3,4] - 1705.888301207771 R[3,5]
   - 1993.053716789376 R[3,6] - 2271.731514844821 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 2559.027367055442 R[3,0] - 2274.319474512749 R[3,1]
   - 1990.552400775131 R[3,2] - 1705.844508201133 R[3,3]
   - 1419.966891342418 R[3,4] - 1135.258998773962 R[3,5]
   - 848.0935831923572 R[3,6] - 569.4157851369125 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_1_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_1_aux_3 = 0
 Profit_Constraint_3: - 600 Sold_0[3] + 600 Lost_0[3] - 600 Sold_1[3]
   + 400 Left_1[3] + 600 Lost_1[3] + profits_vars[3] = 0
 R124: - 19.04682485744599 alphas[0] - 295.975370345291 alphas[2]
   - alphas[3] + f_var[4] = 0
 Q0_upper_bound_4: - 2806.409713277834 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_4_0: - 19.04682485744599 betas[0,0]
   - 295.975370345291 betas[0,2] - betas[0,3] + tau[4,0] = 0
 tau_computation_4_1: - 19.04682485744599 betas[1,0]
   - 295.975370345291 betas[1,2] - betas[1,3] + tau[4,1] = 0
 tau_computation_4_2: - 19.04682485744599 betas[2,0]
   - 295.975370345291 betas[2,2] - betas[2,3] + tau[4,2] = 0
 tau_computation_4_3: - 19.04682485744599 betas[3,0]
   - 295.975370345291 betas[3,2] - betas[3,3] + tau[4,3] = 0
 tau_computation_4_4: - 19.04682485744599 betas[4,0]
   - 295.975370345291 betas[4,2] - betas[4,3] + tau[4,4] = 0
 tau_computation_4_5: - 19.04682485744599 betas[5,0]
   - 295.975370345291 betas[5,2] - betas[5,3] + tau[4,5] = 0
 tau_computation_4_6: - 19.04682485744599 betas[6,0]
   - 295.975370345291 betas[6,2] - betas[6,3] + tau[4,6] = 0
 tau_computation_4_7: - 19.04682485744599 betas[7,0]
   - 295.975370345291 betas[7,2] - betas[7,3] + tau[4,7] = 0
 tau_computation_4_8: - 19.04682485744599 betas[8,0]
   - 295.975370345291 betas[8,2] - betas[8,3] + tau[4,8] = 0
 tau_gap_4_0: tau[4,0] - 5e+06 R[4,0] - tau_star_4 <= -0.001
 tau_gap_4_1: tau[4,1] - 5e+06 R[4,1] - tau_star_4 <= -0.001
 tau_gap_4_2: tau[4,2] - 5e+06 R[4,2] - tau_star_4 <= -0.001
 tau_gap_4_3: tau[4,3] - 5e+06 R[4,3] - tau_star_4 <= -0.001
 tau_gap_4_4: tau[4,4] - 5e+06 R[4,4] - tau_star_4 <= -0.001
 tau_gap_4_5: tau[4,5] - 5e+06 R[4,5] - tau_star_4 <= -0.001
 tau_gap_4_6: tau[4,6] - 5e+06 R[4,6] - tau_star_4 <= -0.001
 tau_gap_4_7: tau[4,7] - 5e+06 R[4,7] - tau_star_4 <= -0.001
 tau_gap_4_8: tau[4,8] - 5e+06 R[4,8] - tau_star_4 <= -0.001
 one_R_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4] + R[4,5] + R[4,6]
   + R[4,7] + R[4,8] = 1
 Define_Q_hat_4: - 2907.962205200044 R[4,0] - 2951.799834530741 R[4,1]
   - 2951.517352400206 R[4,2] - 2951.517352430958 R[4,3]
   - 2956.960548209041 R[4,4] - 2956.960548179745 R[4,5]
   - 2956.587724678847 R[4,6] - 2958.817845534158 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 291.4743411988478 R[4,0] - 587.4497115284614 R[4,1]
   - 883.4703121795069 R[4,2] - 1179.445682526883 R[4,3]
   - 1477.556712217927 R[4,4] - 1773.532082563218 R[4,5]
   - 2069.940457573151 R[4,6] - 2366.867104216971 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 2667.343503708706 R[4,0] - 2371.368133379092 R[4,1]
   - 2075.347532728047 R[4,2] - 1779.372162380671 R[4,3]
   - 1481.261132689626 R[4,4] - 1185.285762344335 R[4,5]
   - 888.8773873344015 R[4,6] - 591.9507406905821 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_1_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_1_aux_4 = 0
 Profit_Constraint_4: - 600 Sold_0[4] + 600 Lost_0[4] - 600 Sold_1[4]
   + 400 Left_1[4] + 600 Lost_1[4] + profits_vars[4] = 0
 R155: - 18.54517730091645 alphas[0] - 285.5227879752246 alphas[2]
   - alphas[3] + f_var[5] = 0
 Q0_upper_bound_5: - 2806.409713277834 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_5_0: - 18.54517730091645 betas[0,0]
   - 285.5227879752246 betas[0,2] - betas[0,3] + tau[5,0] = 0
 tau_computation_5_1: - 18.54517730091645 betas[1,0]
   - 285.5227879752246 betas[1,2] - betas[1,3] + tau[5,1] = 0
 tau_computation_5_2: - 18.54517730091645 betas[2,0]
   - 285.5227879752246 betas[2,2] - betas[2,3] + tau[5,2] = 0
 tau_computation_5_3: - 18.54517730091645 betas[3,0]
   - 285.5227879752246 betas[3,2] - betas[3,3] + tau[5,3] = 0
 tau_computation_5_4: - 18.54517730091645 betas[4,0]
   - 285.5227879752246 betas[4,2] - betas[4,3] + tau[5,4] = 0
 tau_computation_5_5: - 18.54517730091645 betas[5,0]
   - 285.5227879752246 betas[5,2] - betas[5,3] + tau[5,5] = 0
 tau_computation_5_6: - 18.54517730091645 betas[6,0]
   - 285.5227879752246 betas[6,2] - betas[6,3] + tau[5,6] = 0
 tau_computation_5_7: - 18.54517730091645 betas[7,0]
   - 285.5227879752246 betas[7,2] - betas[7,3] + tau[5,7] = 0
 tau_computation_5_8: - 18.54517730091645 betas[8,0]
   - 285.5227879752246 betas[8,2] - betas[8,3] + tau[5,8] = 0
 tau_gap_5_0: tau[5,0] - 5e+06 R[5,0] - tau_star_5 <= -0.001
 tau_gap_5_1: tau[5,1] - 5e+06 R[5,1] - tau_star_5 <= -0.001
 tau_gap_5_2: tau[5,2] - 5e+06 R[5,2] - tau_star_5 <= -0.001
 tau_gap_5_3: tau[5,3] - 5e+06 R[5,3] - tau_star_5 <= -0.001
 tau_gap_5_4: tau[5,4] - 5e+06 R[5,4] - tau_star_5 <= -0.001
 tau_gap_5_5: tau[5,5] - 5e+06 R[5,5] - tau_star_5 <= -0.001
 tau_gap_5_6: tau[5,6] - 5e+06 R[5,6] - tau_star_5 <= -0.001
 tau_gap_5_7: tau[5,7] - 5e+06 R[5,7] - tau_star_5 <= -0.001
 tau_gap_5_8: tau[5,8] - 5e+06 R[5,8] - tau_star_5 <= -0.001
 one_R_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4] + R[5,5] + R[5,6]
   + R[5,7] + R[5,8] = 1
 Define_Q_hat_5: - 2884.478918163617 R[5,0] - 2859.139938566952 R[5,1]
   - 2858.674439538569 R[5,2] - 2858.674439590662 R[5,3]
   - 2856.28063396543 R[5,4] - 2856.280634032021 R[5,5]
   - 2855.55735004883 R[5,6] - 2855.605345304279 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 288.9535560309411 R[5,0] - 574.4763441153209 R[5,1]
   - 864.8218369271121 R[5,2] - 1150.344624894165 R[5,3]
   - 1431.733512056135 R[5,4] - 1717.256300031359 R[5,5]
   - 1999.504356614793 R[5,6] - 2284.559768762125 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 2566.651788681633 R[5,0] - 2281.129000597253 R[5,1]
   - 1990.783507785462 R[5,2] - 1705.260719818409 R[5,3]
   - 1423.871832656439 R[5,4] - 1138.349044681215 R[5,5]
   - 856.1009880977807 R[5,6] - 571.0455759504491 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_1_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_1_aux_5 = 0
 Profit_Constraint_5: - 600 Sold_0[5] + 600 Lost_0[5] - 600 Sold_1[5]
   + 400 Left_1[5] + 600 Lost_1[5] + profits_vars[5] = 0
 R186: - 19.77166907696346 alphas[0] - 258.8502890783748 alphas[2]
   - alphas[3] + f_var[6] = 0
 Q0_upper_bound_6: - 2806.409713277834 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_6_0: - 19.77166907696346 betas[0,0]
   - 258.8502890783748 betas[0,2] - betas[0,3] + tau[6,0] = 0
 tau_computation_6_1: - 19.77166907696346 betas[1,0]
   - 258.8502890783748 betas[1,2] - betas[1,3] + tau[6,1] = 0
 tau_computation_6_2: - 19.77166907696346 betas[2,0]
   - 258.8502890783748 betas[2,2] - betas[2,3] + tau[6,2] = 0
 tau_computation_6_3: - 19.77166907696346 betas[3,0]
   - 258.8502890783748 betas[3,2] - betas[3,3] + tau[6,3] = 0
 tau_computation_6_4: - 19.77166907696346 betas[4,0]
   - 258.8502890783748 betas[4,2] - betas[4,3] + tau[6,4] = 0
 tau_computation_6_5: - 19.77166907696346 betas[5,0]
   - 258.8502890783748 betas[5,2] - betas[5,3] + tau[6,5] = 0
 tau_computation_6_6: - 19.77166907696346 betas[6,0]
   - 258.8502890783748 betas[6,2] - betas[6,3] + tau[6,6] = 0
 tau_computation_6_7: - 19.77166907696346 betas[7,0]
   - 258.8502890783748 betas[7,2] - betas[7,3] + tau[6,7] = 0
 tau_computation_6_8: - 19.77166907696346 betas[8,0]
   - 258.8502890783748 betas[8,2] - betas[8,3] + tau[6,8] = 0
 tau_gap_6_0: tau[6,0] - 5e+06 R[6,0] - tau_star_6 <= -0.001
 tau_gap_6_1: tau[6,1] - 5e+06 R[6,1] - tau_star_6 <= -0.001
 tau_gap_6_2: tau[6,2] - 5e+06 R[6,2] - tau_star_6 <= -0.001
 tau_gap_6_3: tau[6,3] - 5e+06 R[6,3] - tau_star_6 <= -0.001
 tau_gap_6_4: tau[6,4] - 5e+06 R[6,4] - tau_star_6 <= -0.001
 tau_gap_6_5: tau[6,5] - 5e+06 R[6,5] - tau_star_6 <= -0.001
 tau_gap_6_6: tau[6,6] - 5e+06 R[6,6] - tau_star_6 <= -0.001
 tau_gap_6_7: tau[6,7] - 5e+06 R[6,7] - tau_star_6 <= -0.001
 tau_gap_6_8: tau[6,8] - 5e+06 R[6,8] - tau_star_6 <= -0.001
 one_R_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4] + R[6,5] + R[6,6]
   + R[6,7] + R[6,8] = 1
 Define_Q_hat_6: - 2561.056413978961 R[6,0] - 2583.561514428304 R[6,1]
   - 2583.173916863843 R[6,2] - 2583.173917799864 R[6,3]
   - 2584.055207714181 R[6,4] - 2584.055207967187 R[6,5]
   - 2583.639462862124 R[6,6] - 2583.023766105402 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 254.2361567907375 R[6,0] - 513.0864456797038 R[6,1]
   - 773.1050244036232 R[6,2] - 1031.955313481411 R[6,3]
   - 1290.663776562443 R[6,4] - 1549.514065640818 R[6,5]
   - 1806.033199256099 R[6,6] - 2065.32318747511 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 2328.787608841122 R[6,0] - 2069.937319952156 R[6,1]
   - 1809.918741228236 R[6,2] - 1551.068452150448 R[6,3]
   - 1292.359989069417 R[6,4] - 1033.509699991042 R[6,5]
   - 776.9905663757606 R[6,6] - 517.7005781567495 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_1_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_1_aux_6 = 0
 Profit_Constraint_6: - 600 Sold_0[6] + 600 Lost_0[6] - 600 Sold_1[6]
   + 400 Left_1[6] + 600 Lost_1[6] + profits_vars[6] = 0
 R217: - 16.75968120247845 alphas[0] - 274.1759063713729 alphas[2]
   - alphas[3] + f_var[7] = 0
 Q0_upper_bound_7: - 2806.409713277834 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_7_0: - 16.75968120247845 betas[0,0]
   - 274.1759063713729 betas[0,2] - betas[0,3] + tau[7,0] = 0
 tau_computation_7_1: - 16.75968120247845 betas[1,0]
   - 274.1759063713729 betas[1,2] - betas[1,3] + tau[7,1] = 0
 tau_computation_7_2: - 16.75968120247845 betas[2,0]
   - 274.1759063713729 betas[2,2] - betas[2,3] + tau[7,2] = 0
 tau_computation_7_3: - 16.75968120247845 betas[3,0]
   - 274.1759063713729 betas[3,2] - betas[3,3] + tau[7,3] = 0
 tau_computation_7_4: - 16.75968120247845 betas[4,0]
   - 274.1759063713729 betas[4,2] - betas[4,3] + tau[7,4] = 0
 tau_computation_7_5: - 16.75968120247845 betas[5,0]
   - 274.1759063713729 betas[5,2] - betas[5,3] + tau[7,5] = 0
 tau_computation_7_6: - 16.75968120247845 betas[6,0]
   - 274.1759063713729 betas[6,2] - betas[6,3] + tau[7,6] = 0
 tau_computation_7_7: - 16.75968120247845 betas[7,0]
   - 274.1759063713729 betas[7,2] - betas[7,3] + tau[7,7] = 0
 tau_computation_7_8: - 16.75968120247845 betas[8,0]
   - 274.1759063713729 betas[8,2] - betas[8,3] + tau[7,8] = 0
 tau_gap_7_0: tau[7,0] - 5e+06 R[7,0] - tau_star_7 <= -0.001
 tau_gap_7_1: tau[7,1] - 5e+06 R[7,1] - tau_star_7 <= -0.001
 tau_gap_7_2: tau[7,2] - 5e+06 R[7,2] - tau_star_7 <= -0.001
 tau_gap_7_3: tau[7,3] - 5e+06 R[7,3] - tau_star_7 <= -0.001
 tau_gap_7_4: tau[7,4] - 5e+06 R[7,4] - tau_star_7 <= -0.001
 tau_gap_7_5: tau[7,5] - 5e+06 R[7,5] - tau_star_7 <= -0.001
 tau_gap_7_6: tau[7,6] - 5e+06 R[7,6] - tau_star_7 <= -0.001
 tau_gap_7_7: tau[7,7] - 5e+06 R[7,7] - tau_star_7 <= -0.001
 tau_gap_7_8: tau[7,8] - 5e+06 R[7,8] - tau_star_7 <= -0.001
 one_R_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4] + R[7,5] + R[7,6]
   + R[7,7] + R[7,8] = 1
 Define_Q_hat_7: - 2755.015805152552 R[7,0] - 2743.111490234982 R[7,1]
   - 2742.696838189168 R[7,2] - 2742.696838763914 R[7,3]
   - 2747.534979869453 R[7,4] - 2747.534979974094 R[7,5]
   - 2750.092364993919 R[7,6] - 2752.189158211038 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 275.0564940666983 R[7,0] - 549.2324004120692 R[7,1]
   - 826.9029097817918 R[7,2] - 1101.078816167204 R[7,3]
   - 1377.039671963897 R[7,4] - 1651.21557833527 R[7,5]
   - 1926.963673256981 R[7,6] - 2203.837344945745 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 2477.132663621793 R[7,0] - 2202.956757276422 R[7,1]
   - 1925.2862479067 R[7,2] - 1651.110341521288 R[7,3]
   - 1375.149485724594 R[7,4] - 1100.973579353221 R[7,5]
   - 825.2254844315105 R[7,6] - 548.3518127427459 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_1_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_1_aux_7 = 0
 Profit_Constraint_7: - 600 Sold_0[7] + 600 Lost_0[7] - 600 Sold_1[7]
   + 400 Left_1[7] + 600 Lost_1[7] + profits_vars[7] = 0
 R248: - 19.48771382324703 alphas[0] - 257.015800896171 alphas[2]
   - alphas[3] + f_var[8] = 0
 Q0_upper_bound_8: - 2806.409713277834 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_8_0: - 19.48771382324703 betas[0,0]
   - 257.015800896171 betas[0,2] - betas[0,3] + tau[8,0] = 0
 tau_computation_8_1: - 19.48771382324703 betas[1,0]
   - 257.015800896171 betas[1,2] - betas[1,3] + tau[8,1] = 0
 tau_computation_8_2: - 19.48771382324703 betas[2,0]
   - 257.015800896171 betas[2,2] - betas[2,3] + tau[8,2] = 0
 tau_computation_8_3: - 19.48771382324703 betas[3,0]
   - 257.015800896171 betas[3,2] - betas[3,3] + tau[8,3] = 0
 tau_computation_8_4: - 19.48771382324703 betas[4,0]
   - 257.015800896171 betas[4,2] - betas[4,3] + tau[8,4] = 0
 tau_computation_8_5: - 19.48771382324703 betas[5,0]
   - 257.015800896171 betas[5,2] - betas[5,3] + tau[8,5] = 0
 tau_computation_8_6: - 19.48771382324703 betas[6,0]
   - 257.015800896171 betas[6,2] - betas[6,3] + tau[8,6] = 0
 tau_computation_8_7: - 19.48771382324703 betas[7,0]
   - 257.015800896171 betas[7,2] - betas[7,3] + tau[8,7] = 0
 tau_computation_8_8: - 19.48771382324703 betas[8,0]
   - 257.015800896171 betas[8,2] - betas[8,3] + tau[8,8] = 0
 tau_gap_8_0: tau[8,0] - 5e+06 R[8,0] - tau_star_8 <= -0.001
 tau_gap_8_1: tau[8,1] - 5e+06 R[8,1] - tau_star_8 <= -0.001
 tau_gap_8_2: tau[8,2] - 5e+06 R[8,2] - tau_star_8 <= -0.001
 tau_gap_8_3: tau[8,3] - 5e+06 R[8,3] - tau_star_8 <= -0.001
 tau_gap_8_4: tau[8,4] - 5e+06 R[8,4] - tau_star_8 <= -0.001
 tau_gap_8_5: tau[8,5] - 5e+06 R[8,5] - tau_star_8 <= -0.001
 tau_gap_8_6: tau[8,6] - 5e+06 R[8,6] - tau_star_8 <= -0.001
 tau_gap_8_7: tau[8,7] - 5e+06 R[8,7] - tau_star_8 <= -0.001
 tau_gap_8_8: tau[8,8] - 5e+06 R[8,8] - tau_star_8 <= -0.001
 one_R_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4] + R[8,5] + R[8,6]
   + R[8,7] + R[8,8] = 1
 Define_Q_hat_8: - 2619.449080797048 R[8,0] - 2576.58320868366 R[8,1]
   - 2576.318108622771 R[8,2] - 2576.318106324997 R[8,3]
   - 2570.40424213583 R[8,4] - 2570.40424162326 R[8,5]
   - 2578.710130036958 R[8,6] - 2580.751659957173 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 260.5042472927709 R[8,0] - 517.520048811655 R[8,1]
   - 778.1457884737276 R[8,2] - 1035.161589355843 R[8,3]
   - 1287.349947963988 R[8,4] - 1544.365748860159 R[8,5]
   - 1805.14165269625 R[8,6] - 2066.720057398957 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 2320.247411898527 R[8,0] - 2063.231610379644 R[8,1]
   - 1802.605870717571 R[8,2] - 1545.590069835455 R[8,3]
   - 1293.401711227311 R[8,4] - 1036.38591033114 R[8,5]
   - 775.6100064950488 R[8,6] - 514.0316017923419 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_1_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_1_aux_8 = 0
 Profit_Constraint_8: - 600 Sold_0[8] + 600 Lost_0[8] - 600 Sold_1[8]
   + 400 Left_1[8] + 600 Lost_1[8] + profits_vars[8] = 0
 R279: - 18.84983593125044 alphas[0] - 267.9497639169816 alphas[2]
   - alphas[3] + f_var[9] = 0
 Q0_upper_bound_9: - 2806.409713277834 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_9_0: - 18.84983593125044 betas[0,0]
   - 267.9497639169816 betas[0,2] - betas[0,3] + tau[9,0] = 0
 tau_computation_9_1: - 18.84983593125044 betas[1,0]
   - 267.9497639169816 betas[1,2] - betas[1,3] + tau[9,1] = 0
 tau_computation_9_2: - 18.84983593125044 betas[2,0]
   - 267.9497639169816 betas[2,2] - betas[2,3] + tau[9,2] = 0
 tau_computation_9_3: - 18.84983593125044 betas[3,0]
   - 267.9497639169816 betas[3,2] - betas[3,3] + tau[9,3] = 0
 tau_computation_9_4: - 18.84983593125044 betas[4,0]
   - 267.9497639169816 betas[4,2] - betas[4,3] + tau[9,4] = 0
 tau_computation_9_5: - 18.84983593125044 betas[5,0]
   - 267.9497639169816 betas[5,2] - betas[5,3] + tau[9,5] = 0
 tau_computation_9_6: - 18.84983593125044 betas[6,0]
   - 267.9497639169816 betas[6,2] - betas[6,3] + tau[9,6] = 0
 tau_computation_9_7: - 18.84983593125044 betas[7,0]
   - 267.9497639169816 betas[7,2] - betas[7,3] + tau[9,7] = 0
 tau_computation_9_8: - 18.84983593125044 betas[8,0]
   - 267.9497639169816 betas[8,2] - betas[8,3] + tau[9,8] = 0
 tau_gap_9_0: tau[9,0] - 5e+06 R[9,0] - tau_star_9 <= -0.001
 tau_gap_9_1: tau[9,1] - 5e+06 R[9,1] - tau_star_9 <= -0.001
 tau_gap_9_2: tau[9,2] - 5e+06 R[9,2] - tau_star_9 <= -0.001
 tau_gap_9_3: tau[9,3] - 5e+06 R[9,3] - tau_star_9 <= -0.001
 tau_gap_9_4: tau[9,4] - 5e+06 R[9,4] - tau_star_9 <= -0.001
 tau_gap_9_5: tau[9,5] - 5e+06 R[9,5] - tau_star_9 <= -0.001
 tau_gap_9_6: tau[9,6] - 5e+06 R[9,6] - tau_star_9 <= -0.001
 tau_gap_9_7: tau[9,7] - 5e+06 R[9,7] - tau_star_9 <= -0.001
 tau_gap_9_8: tau[9,8] - 5e+06 R[9,8] - tau_star_9 <= -0.001
 one_R_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4] + R[9,5] + R[9,6]
   + R[9,7] + R[9,8] = 1
 Define_Q_hat_9: - 2684.388358292593 R[9,0] - 2679.506487632207 R[9,1]
   - 2679.370872730881 R[9,2] - 2679.370874553195 R[9,3]
   - 2679.586782863303 R[9,4] - 2679.586783399606 R[9,5]
   - 2675.310122596688 R[9,6] - 2675.748107475871 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 267.4750758344236 R[9,0] - 535.4248393303114 R[9,1]
   - 804.251405814184 R[9,2] - 1072.201169711471 R[9,3]
   - 1339.975966173151 R[9,4] - 1607.925730090132 R[9,5]
   - 1871.85094093184 R[9,6] - 2139.848579275315 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 2408.273031274855 R[9,0] - 2140.323267778967 R[9,1]
   - 1871.496701295095 R[9,2] - 1603.546937397808 R[9,3]
   - 1335.772140936128 R[9,4] - 1067.822377019146 R[9,5]
   - 803.8971661774386 R[9,6] - 535.8995278339632 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_1_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_1_aux_9 = 0
 Profit_Constraint_9: - 600 Sold_0[9] + 600 Lost_0[9] - 600 Sold_1[9]
   + 400 Left_1[9] + 600 Lost_1[9] + profits_vars[9] = 0
 R310: - 16.78712325797357 alphas[0] - 296.8558520970259 alphas[2]
   - alphas[3] + f_var[10] = 0
 Q0_upper_bound_10:
   - 2806.409713277834 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 tau_computation_10_0: - 16.78712325797357 betas[0,0]
   - 296.8558520970259 betas[0,2] - betas[0,3] + tau[10,0] = 0
 tau_computation_10_1: - 16.78712325797357 betas[1,0]
   - 296.8558520970259 betas[1,2] - betas[1,3] + tau[10,1] = 0
 tau_computation_10_2: - 16.78712325797357 betas[2,0]
   - 296.8558520970259 betas[2,2] - betas[2,3] + tau[10,2] = 0
 tau_computation_10_3: - 16.78712325797357 betas[3,0]
   - 296.8558520970259 betas[3,2] - betas[3,3] + tau[10,3] = 0
 tau_computation_10_4: - 16.78712325797357 betas[4,0]
   - 296.8558520970259 betas[4,2] - betas[4,3] + tau[10,4] = 0
 tau_computation_10_5: - 16.78712325797357 betas[5,0]
   - 296.8558520970259 betas[5,2] - betas[5,3] + tau[10,5] = 0
 tau_computation_10_6: - 16.78712325797357 betas[6,0]
   - 296.8558520970259 betas[6,2] - betas[6,3] + tau[10,6] = 0
 tau_computation_10_7: - 16.78712325797357 betas[7,0]
   - 296.8558520970259 betas[7,2] - betas[7,3] + tau[10,7] = 0
 tau_computation_10_8: - 16.78712325797357 betas[8,0]
   - 296.8558520970259 betas[8,2] - betas[8,3] + tau[10,8] = 0
 tau_gap_10_0: tau[10,0] - 5e+06 R[10,0] - tau_star_10 <= -0.001
 tau_gap_10_1: tau[10,1] - 5e+06 R[10,1] - tau_star_10 <= -0.001
 tau_gap_10_2: tau[10,2] - 5e+06 R[10,2] - tau_star_10 <= -0.001
 tau_gap_10_3: tau[10,3] - 5e+06 R[10,3] - tau_star_10 <= -0.001
 tau_gap_10_4: tau[10,4] - 5e+06 R[10,4] - tau_star_10 <= -0.001
 tau_gap_10_5: tau[10,5] - 5e+06 R[10,5] - tau_star_10 <= -0.001
 tau_gap_10_6: tau[10,6] - 5e+06 R[10,6] - tau_star_10 <= -0.001
 tau_gap_10_7: tau[10,7] - 5e+06 R[10,7] - tau_star_10 <= -0.001
 tau_gap_10_8: tau[10,8] - 5e+06 R[10,8] - tau_star_10 <= -0.001
 one_R_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4] + R[10,5]
   + R[10,6] + R[10,7] + R[10,8] = 1
 Define_Q_hat_10: - 3010.198182682859 R[10,0] - 2974.494503872428 R[10,1]
   - 2973.953186043129 R[10,2] - 2973.953186576196 R[10,3]
   - 2975.88226550846 R[10,4] - 2975.882265666565 R[10,5]
   - 2976.346896808917 R[10,6] - 2976.405224517583 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 302.4487390872424 R[10,0] - 599.304591221993 R[10,1]
   - 902.3404227617824 R[10,2] - 1199.196274854574 R[10,3]
   - 1494.213455349018 R[10,4] - 1791.069307446044 R[10,5]
   - 2086.747948834848 R[10,6] - 2382.69351978483 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 2673.956484891639 R[10,0] - 2377.100632756889 R[10,1]
   - 2074.0648012171 R[10,2] - 1777.208949124308 R[10,3]
   - 1482.191768629864 R[10,4] - 1185.335916532838 R[10,5]
   - 889.6572751440337 R[10,6] - 593.7117041940518 R[10,7] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: - Total_Demand_Up_to_K_minus_1[10] + Q0_var[10]
   + Lost_0_aux_10 = 0
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_1_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: - Total_Demand_from_k_to_T[10]
   + Q1_plus_left[10] + Lost_1_aux_10 = 0
 Profit_Constraint_10: - 600 Sold_0[10] + 600 Lost_0[10] - 600 Sold_1[10]
   + 400 Left_1[10] + 600 Lost_1[10] + profits_vars[10] = 0
 R341: - 18.10832718226629 alphas[0] - 296.1652653779354 alphas[2]
   - alphas[3] + f_var[11] = 0
 Q0_upper_bound_11:
   - 2806.409713277834 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 tau_computation_11_0: - 18.10832718226629 betas[0,0]
   - 296.1652653779354 betas[0,2] - betas[0,3] + tau[11,0] = 0
 tau_computation_11_1: - 18.10832718226629 betas[1,0]
   - 296.1652653779354 betas[1,2] - betas[1,3] + tau[11,1] = 0
 tau_computation_11_2: - 18.10832718226629 betas[2,0]
   - 296.1652653779354 betas[2,2] - betas[2,3] + tau[11,2] = 0
 tau_computation_11_3: - 18.10832718226629 betas[3,0]
   - 296.1652653779354 betas[3,2] - betas[3,3] + tau[11,3] = 0
 tau_computation_11_4: - 18.10832718226629 betas[4,0]
   - 296.1652653779354 betas[4,2] - betas[4,3] + tau[11,4] = 0
 tau_computation_11_5: - 18.10832718226629 betas[5,0]
   - 296.1652653779354 betas[5,2] - betas[5,3] + tau[11,5] = 0
 tau_computation_11_6: - 18.10832718226629 betas[6,0]
   - 296.1652653779354 betas[6,2] - betas[6,3] + tau[11,6] = 0
 tau_computation_11_7: - 18.10832718226629 betas[7,0]
   - 296.1652653779354 betas[7,2] - betas[7,3] + tau[11,7] = 0
 tau_computation_11_8: - 18.10832718226629 betas[8,0]
   - 296.1652653779354 betas[8,2] - betas[8,3] + tau[11,8] = 0
 tau_gap_11_0: tau[11,0] - 5e+06 R[11,0] - tau_star_11 <= -0.001
 tau_gap_11_1: tau[11,1] - 5e+06 R[11,1] - tau_star_11 <= -0.001
 tau_gap_11_2: tau[11,2] - 5e+06 R[11,2] - tau_star_11 <= -0.001
 tau_gap_11_3: tau[11,3] - 5e+06 R[11,3] - tau_star_11 <= -0.001
 tau_gap_11_4: tau[11,4] - 5e+06 R[11,4] - tau_star_11 <= -0.001
 tau_gap_11_5: tau[11,5] - 5e+06 R[11,5] - tau_star_11 <= -0.001
 tau_gap_11_6: tau[11,6] - 5e+06 R[11,6] - tau_star_11 <= -0.001
 tau_gap_11_7: tau[11,7] - 5e+06 R[11,7] - tau_star_11 <= -0.001
 tau_gap_11_8: tau[11,8] - 5e+06 R[11,8] - tau_star_11 <= -0.001
 one_R_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4] + R[11,5]
   + R[11,6] + R[11,7] + R[11,8] = 1
 Define_Q_hat_11: - 2975.986752263477 R[11,0] - 2963.522244305502 R[11,1]
   - 2963.573783773375 R[11,2] - 2963.573783816621 R[11,3]
   - 2969.820295519373 R[11,4] - 2969.82029551098 R[11,5]
   - 2969.943103821755 R[11,6] - 2967.626251466097 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 298.7763542869255 R[11,0] - 594.9416196565967 R[11,1]
   - 891.5571180479533 R[11,2] - 1187.72238342127 R[11,3]
   - 1487.564999841115 R[11,4] - 1783.73026521905 R[11,5]
   - 2083.027409318757 R[11,6] - 2375.29572005518 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 2668.849896524126 R[11,0] - 2372.684631154455 R[11,1]
   - 2076.069132763098 R[11,2] - 1779.903867389781 R[11,3]
   - 1480.061250969936 R[11,4] - 1183.895985592001 R[11,5]
   - 884.598841492294 R[11,6] - 592.3305307558709 R[11,7] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: - Total_Demand_Up_to_K_minus_1[11] + Q0_var[11]
   + Lost_0_aux_11 = 0
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_1_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: - Total_Demand_from_k_to_T[11]
   + Q1_plus_left[11] + Lost_1_aux_11 = 0
 Profit_Constraint_11: - 600 Sold_0[11] + 600 Lost_0[11] - 600 Sold_1[11]
   + 400 Left_1[11] + 600 Lost_1[11] + profits_vars[11] = 0
 R372: - 16.44284978825848 alphas[0] - 264.1418426088042 alphas[2]
   - alphas[3] + f_var[12] = 0
 Q0_upper_bound_12:
   - 2806.409713277834 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 tau_computation_12_0: - 16.44284978825848 betas[0,0]
   - 264.1418426088042 betas[0,2] - betas[0,3] + tau[12,0] = 0
 tau_computation_12_1: - 16.44284978825848 betas[1,0]
   - 264.1418426088042 betas[1,2] - betas[1,3] + tau[12,1] = 0
 tau_computation_12_2: - 16.44284978825848 betas[2,0]
   - 264.1418426088042 betas[2,2] - betas[2,3] + tau[12,2] = 0
 tau_computation_12_3: - 16.44284978825848 betas[3,0]
   - 264.1418426088042 betas[3,2] - betas[3,3] + tau[12,3] = 0
 tau_computation_12_4: - 16.44284978825848 betas[4,0]
   - 264.1418426088042 betas[4,2] - betas[4,3] + tau[12,4] = 0
 tau_computation_12_5: - 16.44284978825848 betas[5,0]
   - 264.1418426088042 betas[5,2] - betas[5,3] + tau[12,5] = 0
 tau_computation_12_6: - 16.44284978825848 betas[6,0]
   - 264.1418426088042 betas[6,2] - betas[6,3] + tau[12,6] = 0
 tau_computation_12_7: - 16.44284978825848 betas[7,0]
   - 264.1418426088042 betas[7,2] - betas[7,3] + tau[12,7] = 0
 tau_computation_12_8: - 16.44284978825848 betas[8,0]
   - 264.1418426088042 betas[8,2] - betas[8,3] + tau[12,8] = 0
 tau_gap_12_0: tau[12,0] - 5e+06 R[12,0] - tau_star_12 <= -0.001
 tau_gap_12_1: tau[12,1] - 5e+06 R[12,1] - tau_star_12 <= -0.001
 tau_gap_12_2: tau[12,2] - 5e+06 R[12,2] - tau_star_12 <= -0.001
 tau_gap_12_3: tau[12,3] - 5e+06 R[12,3] - tau_star_12 <= -0.001
 tau_gap_12_4: tau[12,4] - 5e+06 R[12,4] - tau_star_12 <= -0.001
 tau_gap_12_5: tau[12,5] - 5e+06 R[12,5] - tau_star_12 <= -0.001
 tau_gap_12_6: tau[12,6] - 5e+06 R[12,6] - tau_star_12 <= -0.001
 tau_gap_12_7: tau[12,7] - 5e+06 R[12,7] - tau_star_12 <= -0.001
 tau_gap_12_8: tau[12,8] - 5e+06 R[12,8] - tau_star_12 <= -0.001
 one_R_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4] + R[12,5]
   + R[12,6] + R[12,7] + R[12,8] = 1
 Define_Q_hat_12: - 2666.449999731956 R[12,0] - 2644.356566316561 R[12,1]
   - 2643.928680348547 R[12,2] - 2643.928680123522 R[12,3]
   - 2640.494747475386 R[12,4] - 2640.494747457255 R[12,5]
   - 2634.446955123482 R[12,6] - 2635.957320601318 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 265.5495043364429 R[12,0] - 529.6913471055822 R[12,1]
   - 797.7468768040378 R[12,2] - 1061.88871941312 R[12,3]
   - 1322.182488467359 R[12,4] - 1586.324331076164 R[12,5]
   - 1842.177017712394 R[12,6] - 2107.673634671482 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 2370.407815552649 R[12,0] - 2106.265972783509 R[12,1]
   - 1838.210443085053 R[12,2] - 1574.068600475971 R[12,3]
   - 1313.774831421732 R[12,4] - 1049.632988812927 R[12,5]
   - 793.7803021766974 R[12,6] - 528.2836852176083 R[12,7] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: - Total_Demand_Up_to_K_minus_1[12] + Q0_var[12]
   + Lost_0_aux_12 = 0
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_1_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: - Total_Demand_from_k_to_T[12]
   + Q1_plus_left[12] + Lost_1_aux_12 = 0
 Profit_Constraint_12: - 600 Sold_0[12] + 600 Lost_0[12] - 600 Sold_1[12]
   + 400 Left_1[12] + 600 Lost_1[12] + profits_vars[12] = 0
 R403: - 19.37199958537421 alphas[0] - 266.9815522083099 alphas[2]
   - alphas[3] + f_var[13] = 0
 Q0_upper_bound_13:
   - 2806.409713277834 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 tau_computation_13_0: - 19.37199958537421 betas[0,0]
   - 266.9815522083099 betas[0,2] - betas[0,3] + tau[13,0] = 0
 tau_computation_13_1: - 19.37199958537421 betas[1,0]
   - 266.9815522083099 betas[1,2] - betas[1,3] + tau[13,1] = 0
 tau_computation_13_2: - 19.37199958537421 betas[2,0]
   - 266.9815522083099 betas[2,2] - betas[2,3] + tau[13,2] = 0
 tau_computation_13_3: - 19.37199958537421 betas[3,0]
   - 266.9815522083099 betas[3,2] - betas[3,3] + tau[13,3] = 0
 tau_computation_13_4: - 19.37199958537421 betas[4,0]
   - 266.9815522083099 betas[4,2] - betas[4,3] + tau[13,4] = 0
 tau_computation_13_5: - 19.37199958537421 betas[5,0]
   - 266.9815522083099 betas[5,2] - betas[5,3] + tau[13,5] = 0
 tau_computation_13_6: - 19.37199958537421 betas[6,0]
   - 266.9815522083099 betas[6,2] - betas[6,3] + tau[13,6] = 0
 tau_computation_13_7: - 19.37199958537421 betas[7,0]
   - 266.9815522083099 betas[7,2] - betas[7,3] + tau[13,7] = 0
 tau_computation_13_8: - 19.37199958537421 betas[8,0]
   - 266.9815522083099 betas[8,2] - betas[8,3] + tau[13,8] = 0
 tau_gap_13_0: tau[13,0] - 5e+06 R[13,0] - tau_star_13 <= -0.001
 tau_gap_13_1: tau[13,1] - 5e+06 R[13,1] - tau_star_13 <= -0.001
 tau_gap_13_2: tau[13,2] - 5e+06 R[13,2] - tau_star_13 <= -0.001
 tau_gap_13_3: tau[13,3] - 5e+06 R[13,3] - tau_star_13 <= -0.001
 tau_gap_13_4: tau[13,4] - 5e+06 R[13,4] - tau_star_13 <= -0.001
 tau_gap_13_5: tau[13,5] - 5e+06 R[13,5] - tau_star_13 <= -0.001
 tau_gap_13_6: tau[13,6] - 5e+06 R[13,6] - tau_star_13 <= -0.001
 tau_gap_13_7: tau[13,7] - 5e+06 R[13,7] - tau_star_13 <= -0.001
 tau_gap_13_8: tau[13,8] - 5e+06 R[13,8] - tau_star_13 <= -0.001
 one_R_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4] + R[13,5]
   + R[13,6] + R[13,7] + R[13,8] = 1
 Define_Q_hat_13: - 2700.50811577473 R[13,0] - 2673.640507002989 R[13,1]
   - 2673.94073514619 R[13,2] - 2673.940735868188 R[13,3]
   - 2673.455362262536 R[13,4] - 2673.455362454726 R[13,5]
   - 2671.63025506185 R[13,6] - 2669.71520048805 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 269.2054317896268 R[13,0] - 536.1869838011164 R[13,1]
   - 801.8618985380269 R[13,2] - 1068.843450752611 R[13,3]
   - 1336.732650817388 R[13,4] - 1603.714203025698 R[13,5]
   - 1871.038175397234 R[13,6] - 2135.752095527041 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 2400.509768154034 R[13,0] - 2133.528216142544 R[13,1]
   - 1867.853301405634 R[13,2] - 1600.87174919105 R[13,3]
   - 1332.982549126273 R[13,4] - 1066.000996917963 R[13,5]
   - 798.6770245464268 R[13,6] - 533.9631044166199 R[13,7] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: - Total_Demand_Up_to_K_minus_1[13] + Q0_var[13]
   + Lost_0_aux_13 = 0
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_1_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: - Total_Demand_from_k_to_T[13]
   + Q1_plus_left[13] + Lost_1_aux_13 = 0
 Profit_Constraint_13: - 600 Sold_0[13] + 600 Lost_0[13] - 600 Sold_1[13]
   + 400 Left_1[13] + 600 Lost_1[13] + profits_vars[13] = 0
 R434: - 15.56213658606156 alphas[0] - 280.0106434065647 alphas[2]
   - alphas[3] + f_var[14] = 0
 Q0_upper_bound_14:
   - 2806.409713277834 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 tau_computation_14_0: - 15.56213658606156 betas[0,0]
   - 280.0106434065647 betas[0,2] - betas[0,3] + tau[14,0] = 0
 tau_computation_14_1: - 15.56213658606156 betas[1,0]
   - 280.0106434065647 betas[1,2] - betas[1,3] + tau[14,1] = 0
 tau_computation_14_2: - 15.56213658606156 betas[2,0]
   - 280.0106434065647 betas[2,2] - betas[2,3] + tau[14,2] = 0
 tau_computation_14_3: - 15.56213658606156 betas[3,0]
   - 280.0106434065647 betas[3,2] - betas[3,3] + tau[14,3] = 0
 tau_computation_14_4: - 15.56213658606156 betas[4,0]
   - 280.0106434065647 betas[4,2] - betas[4,3] + tau[14,4] = 0
 tau_computation_14_5: - 15.56213658606156 betas[5,0]
   - 280.0106434065647 betas[5,2] - betas[5,3] + tau[14,5] = 0
 tau_computation_14_6: - 15.56213658606156 betas[6,0]
   - 280.0106434065647 betas[6,2] - betas[6,3] + tau[14,6] = 0
 tau_computation_14_7: - 15.56213658606156 betas[7,0]
   - 280.0106434065647 betas[7,2] - betas[7,3] + tau[14,7] = 0
 tau_computation_14_8: - 15.56213658606156 betas[8,0]
   - 280.0106434065647 betas[8,2] - betas[8,3] + tau[14,8] = 0
 tau_gap_14_0: tau[14,0] - 5e+06 R[14,0] - tau_star_14 <= -0.001
 tau_gap_14_1: tau[14,1] - 5e+06 R[14,1] - tau_star_14 <= -0.001
 tau_gap_14_2: tau[14,2] - 5e+06 R[14,2] - tau_star_14 <= -0.001
 tau_gap_14_3: tau[14,3] - 5e+06 R[14,3] - tau_star_14 <= -0.001
 tau_gap_14_4: tau[14,4] - 5e+06 R[14,4] - tau_star_14 <= -0.001
 tau_gap_14_5: tau[14,5] - 5e+06 R[14,5] - tau_star_14 <= -0.001
 tau_gap_14_6: tau[14,6] - 5e+06 R[14,6] - tau_star_14 <= -0.001
 tau_gap_14_7: tau[14,7] - 5e+06 R[14,7] - tau_star_14 <= -0.001
 tau_gap_14_8: tau[14,8] - 5e+06 R[14,8] - tau_star_14 <= -0.001
 one_R_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4] + R[14,5]
   + R[14,6] + R[14,7] + R[14,8] = 1
 Define_Q_hat_14: - 2783.91282400418 R[14,0] - 2797.179932543417 R[14,1]
   - 2797.295333378022 R[14,2] - 2797.295333124269 R[14,3]
   - 2798.10606863422 R[14,4] - 2798.106068586747 R[14,5]
   - 2799.784547201597 R[14,6] - 2799.72884822705 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 278.1584097569896 R[14,0] - 558.1690531612101 R[14,1]
   - 836.4592690079419 R[14,2] - 1116.469912406302 R[14,3]
   - 1397.297453188086 R[14,4] - 1677.30809659465 R[14,5]
   - 1960.099216788134 R[14,6] - 2239.707560804969 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 2521.570437861109 R[14,0] - 2241.559794456888 R[14,1]
   - 1963.269578610156 R[14,2] - 1683.258935211797 R[14,3]
   - 1402.431394430013 R[14,4] - 1122.420751023448 R[14,5]
   - 839.6296308299645 R[14,6] - 560.0212868131293 R[14,7] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: - Total_Demand_Up_to_K_minus_1[14] + Q0_var[14]
   + Lost_0_aux_14 = 0
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_1_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: - Total_Demand_from_k_to_T[14]
   + Q1_plus_left[14] + Lost_1_aux_14 = 0
 Profit_Constraint_14: - 600 Sold_0[14] + 600 Lost_0[14] - 600 Sold_1[14]
   + 400 Left_1[14] + 600 Lost_1[14] + profits_vars[14] = 0
 R465: - 16.0621718064702 alphas[0] - 298.159864763019 alphas[2]
   - alphas[3] + f_var[15] = 0
 Q0_upper_bound_15:
   - 2806.409713277834 Fraction_for_second_order_amount[15] + Q0_var[15]
   = 0
 tau_computation_15_0: - 16.0621718064702 betas[0,0]
   - 298.159864763019 betas[0,2] - betas[0,3] + tau[15,0] = 0
 tau_computation_15_1: - 16.0621718064702 betas[1,0]
   - 298.159864763019 betas[1,2] - betas[1,3] + tau[15,1] = 0
 tau_computation_15_2: - 16.0621718064702 betas[2,0]
   - 298.159864763019 betas[2,2] - betas[2,3] + tau[15,2] = 0
 tau_computation_15_3: - 16.0621718064702 betas[3,0]
   - 298.159864763019 betas[3,2] - betas[3,3] + tau[15,3] = 0
 tau_computation_15_4: - 16.0621718064702 betas[4,0]
   - 298.159864763019 betas[4,2] - betas[4,3] + tau[15,4] = 0
 tau_computation_15_5: - 16.0621718064702 betas[5,0]
   - 298.159864763019 betas[5,2] - betas[5,3] + tau[15,5] = 0
 tau_computation_15_6: - 16.0621718064702 betas[6,0]
   - 298.159864763019 betas[6,2] - betas[6,3] + tau[15,6] = 0
 tau_computation_15_7: - 16.0621718064702 betas[7,0]
   - 298.159864763019 betas[7,2] - betas[7,3] + tau[15,7] = 0
 tau_computation_15_8: - 16.0621718064702 betas[8,0]
   - 298.159864763019 betas[8,2] - betas[8,3] + tau[15,8] = 0
 tau_gap_15_0: tau[15,0] - 5e+06 R[15,0] - tau_star_15 <= -0.001
 tau_gap_15_1: tau[15,1] - 5e+06 R[15,1] - tau_star_15 <= -0.001
 tau_gap_15_2: tau[15,2] - 5e+06 R[15,2] - tau_star_15 <= -0.001
 tau_gap_15_3: tau[15,3] - 5e+06 R[15,3] - tau_star_15 <= -0.001
 tau_gap_15_4: tau[15,4] - 5e+06 R[15,4] - tau_star_15 <= -0.001
 tau_gap_15_5: tau[15,5] - 5e+06 R[15,5] - tau_star_15 <= -0.001
 tau_gap_15_6: tau[15,6] - 5e+06 R[15,6] - tau_star_15 <= -0.001
 tau_gap_15_7: tau[15,7] - 5e+06 R[15,7] - tau_star_15 <= -0.001
 tau_gap_15_8: tau[15,8] - 5e+06 R[15,8] - tau_star_15 <= -0.001
 one_R_15: R[15,0] + R[15,1] + R[15,2] + R[15,3] + R[15,4] + R[15,5]
   + R[15,6] + R[15,7] + R[15,8] = 1
 Define_Q_hat_15: - 3010.476592848339 R[15,0] - 2985.660556084606 R[15,1]
   - 2986.162240875705 R[15,2] - 2986.162240705334 R[15,3]
   - 2983.270156685083 R[15,4] - 2983.270156627886 R[15,5]
   - 2986.207388058533 R[15,6] - 2987.137052854184 R[15,7] + Q_hat[15] = 0
 Adjust_Q_hat_15: Q0_var[15] - Q_hat[15] + Q_hat_adjusted[15] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_15: Total_Demand_Up_to_K_minus_1[15]
   - 302.478624691189 R[15,0] - 600.6384894436436 R[15,1]
   - 896.5167480387029 R[15,2] - 1194.676612819091 R[15,3]
   - 1491.527101725558 R[15,4] - 1789.686966488577 R[15,5]
   - 2093.311165115699 R[15,6] - 2390.817322744038 R[15,7] = 0
 Constr_Total_Demand_from_K_to_T_15: Total_Demand_from_k_to_T[15]
   - 2684.658427578887 R[15,0] - 2386.498562826433 R[15,1]
   - 2090.620304231374 R[15,2] - 1792.460439450985 R[15,3]
   - 1495.609950544519 R[15,4] - 1197.4500857815 R[15,5]
   - 893.8258871543776 R[15,6] - 596.319729526038 R[15,7] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: - Total_Demand_Up_to_K_minus_1[15] + Q0_var[15]
   + Lost_0_aux_15 = 0
 Constr_Q1_plus_left_15: - Left_0[15] + Q1_plus_left[15] - Q1_var[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_1_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: - Total_Demand_from_k_to_T[15]
   + Q1_plus_left[15] + Lost_1_aux_15 = 0
 Profit_Constraint_15: - 600 Sold_0[15] + 600 Lost_0[15] - 600 Sold_1[15]
   + 400 Left_1[15] + 600 Lost_1[15] + profits_vars[15] = 0
 R496: - 15.91516646039961 alphas[0] - 257.3900667032695 alphas[2]
   - alphas[3] + f_var[16] = 0
 Q0_upper_bound_16:
   - 2806.409713277834 Fraction_for_second_order_amount[16] + Q0_var[16]
   = 0
 tau_computation_16_0: - 15.91516646039961 betas[0,0]
   - 257.3900667032695 betas[0,2] - betas[0,3] + tau[16,0] = 0
 tau_computation_16_1: - 15.91516646039961 betas[1,0]
   - 257.3900667032695 betas[1,2] - betas[1,3] + tau[16,1] = 0
 tau_computation_16_2: - 15.91516646039961 betas[2,0]
   - 257.3900667032695 betas[2,2] - betas[2,3] + tau[16,2] = 0
 tau_computation_16_3: - 15.91516646039961 betas[3,0]
   - 257.3900667032695 betas[3,2] - betas[3,3] + tau[16,3] = 0
 tau_computation_16_4: - 15.91516646039961 betas[4,0]
   - 257.3900667032695 betas[4,2] - betas[4,3] + tau[16,4] = 0
 tau_computation_16_5: - 15.91516646039961 betas[5,0]
   - 257.3900667032695 betas[5,2] - betas[5,3] + tau[16,5] = 0
 tau_computation_16_6: - 15.91516646039961 betas[6,0]
   - 257.3900667032695 betas[6,2] - betas[6,3] + tau[16,6] = 0
 tau_computation_16_7: - 15.91516646039961 betas[7,0]
   - 257.3900667032695 betas[7,2] - betas[7,3] + tau[16,7] = 0
 tau_computation_16_8: - 15.91516646039961 betas[8,0]
   - 257.3900667032695 betas[8,2] - betas[8,3] + tau[16,8] = 0
 tau_gap_16_0: tau[16,0] - 5e+06 R[16,0] - tau_star_16 <= -0.001
 tau_gap_16_1: tau[16,1] - 5e+06 R[16,1] - tau_star_16 <= -0.001
 tau_gap_16_2: tau[16,2] - 5e+06 R[16,2] - tau_star_16 <= -0.001
 tau_gap_16_3: tau[16,3] - 5e+06 R[16,3] - tau_star_16 <= -0.001
 tau_gap_16_4: tau[16,4] - 5e+06 R[16,4] - tau_star_16 <= -0.001
 tau_gap_16_5: tau[16,5] - 5e+06 R[16,5] - tau_star_16 <= -0.001
 tau_gap_16_6: tau[16,6] - 5e+06 R[16,6] - tau_star_16 <= -0.001
 tau_gap_16_7: tau[16,7] - 5e+06 R[16,7] - tau_star_16 <= -0.001
 tau_gap_16_8: tau[16,8] - 5e+06 R[16,8] - tau_star_16 <= -0.001
 one_R_16: R[16,0] + R[16,1] + R[16,2] + R[16,3] + R[16,4] + R[16,5]
   + R[16,6] + R[16,7] + R[16,8] = 1
 Define_Q_hat_16: - 2584.936610470806 R[16,0] - 2574.650633986202 R[16,1]
   - 2574.852520386185 R[16,2] - 2574.852520761206 R[16,3]
   - 2577.874698063495 R[16,4] - 2577.874698164903 R[16,5]
   - 2579.354787980125 R[16,6] - 2577.215196511934 R[16,7] + Q_hat[16] = 0
 Adjust_Q_hat_16: Q0_var[16] - Q_hat[16] + Q_hat_adjusted[16] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_16: Total_Demand_Up_to_K_minus_1[16]
   - 256.7995477261701 R[16,0] - 514.189614293272 R[16,1]
   - 769.9510425627384 R[16,2] - 1027.341109261272 R[16,3]
   - 1288.260417729404 R[16,4] - 1545.650484432674 R[16,5]
   - 1806.339124186922 R[16,6] - 2062.435062552 R[16,7] = 0
 Constr_Total_Demand_from_K_to_T_16: Total_Demand_from_k_to_T[16]
   - 2320.415648232369 R[16,0] - 2063.025581665267 R[16,1]
   - 1807.264153395801 R[16,2] - 1549.874086697267 R[16,3]
   - 1288.954778229135 R[16,4] - 1031.564711525865 R[16,5]
   - 770.8760717716168 R[16,6] - 514.780133406539 R[16,7] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: - Total_Demand_Up_to_K_minus_1[16] + Q0_var[16]
   + Lost_0_aux_16 = 0
 Constr_Q1_plus_left_16: - Left_0[16] + Q1_plus_left[16] - Q1_var[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_1_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: - Total_Demand_from_k_to_T[16]
   + Q1_plus_left[16] + Lost_1_aux_16 = 0
 Profit_Constraint_16: - 600 Sold_0[16] + 600 Lost_0[16] - 600 Sold_1[16]
   + 400 Left_1[16] + 600 Lost_1[16] + profits_vars[16] = 0
 R527: - 17.01513001202144 alphas[0] - 262.8458321843335 alphas[2]
   - alphas[3] + f_var[17] = 0
 Q0_upper_bound_17:
   - 2806.409713277834 Fraction_for_second_order_amount[17] + Q0_var[17]
   = 0
 tau_computation_17_0: - 17.01513001202144 betas[0,0]
   - 262.8458321843335 betas[0,2] - betas[0,3] + tau[17,0] = 0
 tau_computation_17_1: - 17.01513001202144 betas[1,0]
   - 262.8458321843335 betas[1,2] - betas[1,3] + tau[17,1] = 0
 tau_computation_17_2: - 17.01513001202144 betas[2,0]
   - 262.8458321843335 betas[2,2] - betas[2,3] + tau[17,2] = 0
 tau_computation_17_3: - 17.01513001202144 betas[3,0]
   - 262.8458321843335 betas[3,2] - betas[3,3] + tau[17,3] = 0
 tau_computation_17_4: - 17.01513001202144 betas[4,0]
   - 262.8458321843335 betas[4,2] - betas[4,3] + tau[17,4] = 0
 tau_computation_17_5: - 17.01513001202144 betas[5,0]
   - 262.8458321843335 betas[5,2] - betas[5,3] + tau[17,5] = 0
 tau_computation_17_6: - 17.01513001202144 betas[6,0]
   - 262.8458321843335 betas[6,2] - betas[6,3] + tau[17,6] = 0
 tau_computation_17_7: - 17.01513001202144 betas[7,0]
   - 262.8458321843335 betas[7,2] - betas[7,3] + tau[17,7] = 0
 tau_computation_17_8: - 17.01513001202144 betas[8,0]
   - 262.8458321843335 betas[8,2] - betas[8,3] + tau[17,8] = 0
 tau_gap_17_0: tau[17,0] - 5e+06 R[17,0] - tau_star_17 <= -0.001
 tau_gap_17_1: tau[17,1] - 5e+06 R[17,1] - tau_star_17 <= -0.001
 tau_gap_17_2: tau[17,2] - 5e+06 R[17,2] - tau_star_17 <= -0.001
 tau_gap_17_3: tau[17,3] - 5e+06 R[17,3] - tau_star_17 <= -0.001
 tau_gap_17_4: tau[17,4] - 5e+06 R[17,4] - tau_star_17 <= -0.001
 tau_gap_17_5: tau[17,5] - 5e+06 R[17,5] - tau_star_17 <= -0.001
 tau_gap_17_6: tau[17,6] - 5e+06 R[17,6] - tau_star_17 <= -0.001
 tau_gap_17_7: tau[17,7] - 5e+06 R[17,7] - tau_star_17 <= -0.001
 tau_gap_17_8: tau[17,8] - 5e+06 R[17,8] - tau_star_17 <= -0.001
 one_R_17: R[17,0] + R[17,1] + R[17,2] + R[17,3] + R[17,4] + R[17,5]
   + R[17,6] + R[17,7] + R[17,8] = 1
 Define_Q_hat_17: - 2577.746617822235 R[17,0] - 2620.126701578723 R[17,1]
   - 2620.26739795174 R[17,2] - 2620.267398833225 R[17,3]
   - 2619.412677756326 R[17,4] - 2619.412677982632 R[17,5]
   - 2615.937274472687 R[17,6] - 2619.239818637139 R[17,7] + Q_hat[17] = 0
 Adjust_Q_hat_17: Q0_var[17] - Q_hat[17] + Q_hat_adjusted[17] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_17: Total_Demand_Up_to_K_minus_1[17]
   - 256.0277466236345 R[17,0] - 518.8735784986741 R[17,1]
   - 777.9233307735702 R[17,2] - 1040.769162963633 R[17,3]
   - 1304.189273733873 R[17,4] - 1567.035105918206 R[17,5]
   - 1827.571603810241 R[17,6] - 2093.548153776585 R[17,7] = 0
 Constr_Total_Demand_from_K_to_T_17: Total_Demand_from_k_to_T[17]
   - 2363.212071521617 R[17,0] - 2100.366239646577 R[17,1]
   - 1841.316487371681 R[17,2] - 1578.470655181618 R[17,3]
   - 1315.050544411379 R[17,4] - 1052.204712227045 R[17,5]
   - 791.6682143350106 R[17,6] - 525.6916643686669 R[17,7] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: - Total_Demand_Up_to_K_minus_1[17] + Q0_var[17]
   + Lost_0_aux_17 = 0
 Constr_Q1_plus_left_17: - Left_0[17] + Q1_plus_left[17] - Q1_var[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_1_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: - Total_Demand_from_k_to_T[17]
   + Q1_plus_left[17] + Lost_1_aux_17 = 0
 Profit_Constraint_17: - 600 Sold_0[17] + 600 Lost_0[17] - 600 Sold_1[17]
   + 400 Left_1[17] + 600 Lost_1[17] + profits_vars[17] = 0
 R558: - 18.72616480016065 alphas[0] - 293.6778413645386 alphas[2]
   - alphas[3] + f_var[18] = 0
 Q0_upper_bound_18:
   - 2806.409713277834 Fraction_for_second_order_amount[18] + Q0_var[18]
   = 0
 tau_computation_18_0: - 18.72616480016065 betas[0,0]
   - 293.6778413645386 betas[0,2] - betas[0,3] + tau[18,0] = 0
 tau_computation_18_1: - 18.72616480016065 betas[1,0]
   - 293.6778413645386 betas[1,2] - betas[1,3] + tau[18,1] = 0
 tau_computation_18_2: - 18.72616480016065 betas[2,0]
   - 293.6778413645386 betas[2,2] - betas[2,3] + tau[18,2] = 0
 tau_computation_18_3: - 18.72616480016065 betas[3,0]
   - 293.6778413645386 betas[3,2] - betas[3,3] + tau[18,3] = 0
 tau_computation_18_4: - 18.72616480016065 betas[4,0]
   - 293.6778413645386 betas[4,2] - betas[4,3] + tau[18,4] = 0
 tau_computation_18_5: - 18.72616480016065 betas[5,0]
   - 293.6778413645386 betas[5,2] - betas[5,3] + tau[18,5] = 0
 tau_computation_18_6: - 18.72616480016065 betas[6,0]
   - 293.6778413645386 betas[6,2] - betas[6,3] + tau[18,6] = 0
 tau_computation_18_7: - 18.72616480016065 betas[7,0]
   - 293.6778413645386 betas[7,2] - betas[7,3] + tau[18,7] = 0
 tau_computation_18_8: - 18.72616480016065 betas[8,0]
   - 293.6778413645386 betas[8,2] - betas[8,3] + tau[18,8] = 0
 tau_gap_18_0: tau[18,0] - 5e+06 R[18,0] - tau_star_18 <= -0.001
 tau_gap_18_1: tau[18,1] - 5e+06 R[18,1] - tau_star_18 <= -0.001
 tau_gap_18_2: tau[18,2] - 5e+06 R[18,2] - tau_star_18 <= -0.001
 tau_gap_18_3: tau[18,3] - 5e+06 R[18,3] - tau_star_18 <= -0.001
 tau_gap_18_4: tau[18,4] - 5e+06 R[18,4] - tau_star_18 <= -0.001
 tau_gap_18_5: tau[18,5] - 5e+06 R[18,5] - tau_star_18 <= -0.001
 tau_gap_18_6: tau[18,6] - 5e+06 R[18,6] - tau_star_18 <= -0.001
 tau_gap_18_7: tau[18,7] - 5e+06 R[18,7] - tau_star_18 <= -0.001
 tau_gap_18_8: tau[18,8] - 5e+06 R[18,8] - tau_star_18 <= -0.001
 one_R_18: R[18,0] + R[18,1] + R[18,2] + R[18,3] + R[18,4] + R[18,5]
   + R[18,6] + R[18,7] + R[18,8] = 1
 Define_Q_hat_18: - 2936.898137659225 R[18,0] - 2936.496611848498 R[18,1]
   - 2936.509104365651 R[18,2] - 2936.50910435382 R[18,3]
   - 2931.375578738428 R[18,4] - 2931.375578752609 R[18,5]
   - 2931.115264822964 R[18,6] - 2930.166860966703 R[18,7] + Q_hat[18] = 0
 Adjust_Q_hat_18: Q0_var[18] - Q_hat[18] + Q_hat_adjusted[18] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_18: Total_Demand_Up_to_K_minus_1[18]
   - 294.5804340234814 R[18,0] - 588.2582753966583 R[18,1]
   - 882.0006546647683 R[18,2] - 1175.678496040222 R[18,3]
   - 1464.883849563048 R[18,4] - 1758.561690927586 R[18,5]
   - 2051.420927871104 R[18,6] - 2342.811177641892 R[18,7] = 0
 Constr_Total_Demand_from_K_to_T_18: Total_Demand_from_k_to_T[18]
   - 2635.586426347487 R[18,0] - 2341.908584974311 R[18,1]
   - 2048.166205706201 R[18,2] - 1754.488364330747 R[18,3]
   - 1465.283010807921 R[18,4] - 1171.605169443383 R[18,5]
   - 878.7459324998649 R[18,6] - 587.3556827290771 R[18,7] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: - Total_Demand_Up_to_K_minus_1[18] + Q0_var[18]
   + Lost_0_aux_18 = 0
 Constr_Q1_plus_left_18: - Left_0[18] + Q1_plus_left[18] - Q1_var[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_1_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: - Total_Demand_from_k_to_T[18]
   + Q1_plus_left[18] + Lost_1_aux_18 = 0
 Profit_Constraint_18: - 600 Sold_0[18] + 600 Lost_0[18] - 600 Sold_1[18]
   + 400 Left_1[18] + 600 Lost_1[18] + profits_vars[18] = 0
 R589: - 17.6345372452609 alphas[0] - 274.5946115854172 alphas[2]
   - alphas[3] + f_var[19] = 0
 Q0_upper_bound_19:
   - 2806.409713277834 Fraction_for_second_order_amount[19] + Q0_var[19]
   = 0
 tau_computation_19_0: - 17.6345372452609 betas[0,0]
   - 274.5946115854172 betas[0,2] - betas[0,3] + tau[19,0] = 0
 tau_computation_19_1: - 17.6345372452609 betas[1,0]
   - 274.5946115854172 betas[1,2] - betas[1,3] + tau[19,1] = 0
 tau_computation_19_2: - 17.6345372452609 betas[2,0]
   - 274.5946115854172 betas[2,2] - betas[2,3] + tau[19,2] = 0
 tau_computation_19_3: - 17.6345372452609 betas[3,0]
   - 274.5946115854172 betas[3,2] - betas[3,3] + tau[19,3] = 0
 tau_computation_19_4: - 17.6345372452609 betas[4,0]
   - 274.5946115854172 betas[4,2] - betas[4,3] + tau[19,4] = 0
 tau_computation_19_5: - 17.6345372452609 betas[5,0]
   - 274.5946115854172 betas[5,2] - betas[5,3] + tau[19,5] = 0
 tau_computation_19_6: - 17.6345372452609 betas[6,0]
   - 274.5946115854172 betas[6,2] - betas[6,3] + tau[19,6] = 0
 tau_computation_19_7: - 17.6345372452609 betas[7,0]
   - 274.5946115854172 betas[7,2] - betas[7,3] + tau[19,7] = 0
 tau_computation_19_8: - 17.6345372452609 betas[8,0]
   - 274.5946115854172 betas[8,2] - betas[8,3] + tau[19,8] = 0
 tau_gap_19_0: tau[19,0] - 5e+06 R[19,0] - tau_star_19 <= -0.001
 tau_gap_19_1: tau[19,1] - 5e+06 R[19,1] - tau_star_19 <= -0.001
 tau_gap_19_2: tau[19,2] - 5e+06 R[19,2] - tau_star_19 <= -0.001
 tau_gap_19_3: tau[19,3] - 5e+06 R[19,3] - tau_star_19 <= -0.001
 tau_gap_19_4: tau[19,4] - 5e+06 R[19,4] - tau_star_19 <= -0.001
 tau_gap_19_5: tau[19,5] - 5e+06 R[19,5] - tau_star_19 <= -0.001
 tau_gap_19_6: tau[19,6] - 5e+06 R[19,6] - tau_star_19 <= -0.001
 tau_gap_19_7: tau[19,7] - 5e+06 R[19,7] - tau_star_19 <= -0.001
 tau_gap_19_8: tau[19,8] - 5e+06 R[19,8] - tau_star_19 <= -0.001
 one_R_19: R[19,0] + R[19,1] + R[19,2] + R[19,3] + R[19,4] + R[19,5]
   + R[19,6] + R[19,7] + R[19,8] = 1
 Define_Q_hat_19: - 2785.866761788167 R[19,0] - 2751.265224473598 R[19,1]
   - 2751.002055441513 R[19,2] - 2751.002056194695 R[19,3]
   - 2752.740417971911 R[19,4] - 2752.7404181867 R[19,5]
   - 2752.771937254687 R[19,6] - 2750.520254367741 R[19,7] + Q_hat[19] = 0
 Adjust_Q_hat_19: Q0_var[19] - Q_hat[19] + Q_hat_adjusted[19] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_19: Total_Demand_Up_to_K_minus_1[19]
   - 278.3681528554428 R[19,0] - 552.962764359221 R[19,1]
   - 831.094711946351 R[19,2] - 1105.689323528134 R[19,3]
   - 1380.2837456752 R[19,4] - 1654.878357260618 R[19,5]
   - 1928.968040966164 R[19,6] - 2201.331030674713 R[19,7] = 0
 Constr_Total_Demand_from_K_to_T_19: Total_Demand_from_k_to_T[19]
   - 2472.152100990105 R[19,0] - 2197.557489486327 R[19,1]
   - 1919.425541899196 R[19,2] - 1644.830930317413 R[19,3]
   - 1370.236508170347 R[19,4] - 1095.64189658493 R[19,5]
   - 821.5522128793832 R[19,6] - 549.1892231708345 R[19,7] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: - Total_Demand_Up_to_K_minus_1[19] + Q0_var[19]
   + Lost_0_aux_19 = 0
 Constr_Q1_plus_left_19: - Left_0[19] + Q1_plus_left[19] - Q1_var[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_1_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: - Total_Demand_from_k_to_T[19]
   + Q1_plus_left[19] + Lost_1_aux_19 = 0
 Profit_Constraint_19: - 600 Sold_0[19] + 600 Lost_0[19] - 600 Sold_1[19]
   + 400 Left_1[19] + 600 Lost_1[19] + profits_vars[19] = 0
 R620: - 17.43838161769104 alphas[0] - 294.9480546113516 alphas[2]
   - alphas[3] + f_var[20] = 0
 Q0_upper_bound_20:
   - 2806.409713277834 Fraction_for_second_order_amount[20] + Q0_var[20]
   = 0
 tau_computation_20_0: - 17.43838161769104 betas[0,0]
   - 294.9480546113516 betas[0,2] - betas[0,3] + tau[20,0] = 0
 tau_computation_20_1: - 17.43838161769104 betas[1,0]
   - 294.9480546113516 betas[1,2] - betas[1,3] + tau[20,1] = 0
 tau_computation_20_2: - 17.43838161769104 betas[2,0]
   - 294.9480546113516 betas[2,2] - betas[2,3] + tau[20,2] = 0
 tau_computation_20_3: - 17.43838161769104 betas[3,0]
   - 294.9480546113516 betas[3,2] - betas[3,3] + tau[20,3] = 0
 tau_computation_20_4: - 17.43838161769104 betas[4,0]
   - 294.9480546113516 betas[4,2] - betas[4,3] + tau[20,4] = 0
 tau_computation_20_5: - 17.43838161769104 betas[5,0]
   - 294.9480546113516 betas[5,2] - betas[5,3] + tau[20,5] = 0
 tau_computation_20_6: - 17.43838161769104 betas[6,0]
   - 294.9480546113516 betas[6,2] - betas[6,3] + tau[20,6] = 0
 tau_computation_20_7: - 17.43838161769104 betas[7,0]
   - 294.9480546113516 betas[7,2] - betas[7,3] + tau[20,7] = 0
 tau_computation_20_8: - 17.43838161769104 betas[8,0]
   - 294.9480546113516 betas[8,2] - betas[8,3] + tau[20,8] = 0
 tau_gap_20_0: tau[20,0] - 5e+06 R[20,0] - tau_star_20 <= -0.001
 tau_gap_20_1: tau[20,1] - 5e+06 R[20,1] - tau_star_20 <= -0.001
 tau_gap_20_2: tau[20,2] - 5e+06 R[20,2] - tau_star_20 <= -0.001
 tau_gap_20_3: tau[20,3] - 5e+06 R[20,3] - tau_star_20 <= -0.001
 tau_gap_20_4: tau[20,4] - 5e+06 R[20,4] - tau_star_20 <= -0.001
 tau_gap_20_5: tau[20,5] - 5e+06 R[20,5] - tau_star_20 <= -0.001
 tau_gap_20_6: tau[20,6] - 5e+06 R[20,6] - tau_star_20 <= -0.001
 tau_gap_20_7: tau[20,7] - 5e+06 R[20,7] - tau_star_20 <= -0.001
 tau_gap_20_8: tau[20,8] - 5e+06 R[20,8] - tau_star_20 <= -0.001
 one_R_20: R[20,0] + R[20,1] + R[20,2] + R[20,3] + R[20,4] + R[20,5]
   + R[20,6] + R[20,7] + R[20,8] = 1
 Define_Q_hat_20: - 2947.002329311118 R[20,0] - 2948.833553172868 R[20,1]
   - 2949.006311894861 R[20,2] - 2949.006311722371 R[20,3]
   - 2945.473580476161 R[20,4] - 2945.473580456151 R[20,5]
   - 2947.701561540801 R[20,6] - 2949.394140263179 R[20,7] + Q_hat[20] = 0
 Adjust_Q_hat_20: Q0_var[20] - Q_hat[20] + Q_hat_adjusted[20] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_20: Total_Demand_Up_to_K_minus_1[20]
   - 295.6650563086949 R[20,0] - 590.6131109206378 R[20,1]
   - 884.3190189453928 R[20,2] - 1179.267073558554 R[20,3]
   - 1471.463858242527 R[20,4] - 1766.411912853879 R[20,5]
   - 2064.024328968129 R[20,6] - 2359.49803047933 R[20,7] = 0
 Constr_Total_Demand_from_K_to_T_20: Total_Demand_from_k_to_T[20]
   - 2653.729083393338 R[20,0] - 2358.781028781395 R[20,1]
   - 2065.07512075664 R[20,2] - 1770.127066143479 R[20,3]
   - 1477.930281459506 R[20,4] - 1182.982226848154 R[20,5]
   - 885.3698107339033 R[20,6] - 589.8961092227032 R[20,7] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: - Total_Demand_Up_to_K_minus_1[20] + Q0_var[20]
   + Lost_0_aux_20 = 0
 Constr_Q1_plus_left_20: - Left_0[20] + Q1_plus_left[20] - Q1_var[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_1_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: - Total_Demand_from_k_to_T[20]
   + Q1_plus_left[20] + Lost_1_aux_20 = 0
 Profit_Constraint_20: - 600 Sold_0[20] + 600 Lost_0[20] - 600 Sold_1[20]
   + 400 Left_1[20] + 600 Lost_1[20] + profits_vars[20] = 0
 R651: - 15.00272982448498 alphas[0] - 259.2758948761588 alphas[2]
   - alphas[3] + f_var[21] = 0
 Q0_upper_bound_21:
   - 2806.409713277834 Fraction_for_second_order_amount[21] + Q0_var[21]
   = 0
 tau_computation_21_0: - 15.00272982448498 betas[0,0]
   - 259.2758948761588 betas[0,2] - betas[0,3] + tau[21,0] = 0
 tau_computation_21_1: - 15.00272982448498 betas[1,0]
   - 259.2758948761588 betas[1,2] - betas[1,3] + tau[21,1] = 0
 tau_computation_21_2: - 15.00272982448498 betas[2,0]
   - 259.2758948761588 betas[2,2] - betas[2,3] + tau[21,2] = 0
 tau_computation_21_3: - 15.00272982448498 betas[3,0]
   - 259.2758948761588 betas[3,2] - betas[3,3] + tau[21,3] = 0
 tau_computation_21_4: - 15.00272982448498 betas[4,0]
   - 259.2758948761588 betas[4,2] - betas[4,3] + tau[21,4] = 0
 tau_computation_21_5: - 15.00272982448498 betas[5,0]
   - 259.2758948761588 betas[5,2] - betas[5,3] + tau[21,5] = 0
 tau_computation_21_6: - 15.00272982448498 betas[6,0]
   - 259.2758948761588 betas[6,2] - betas[6,3] + tau[21,6] = 0
 tau_computation_21_7: - 15.00272982448498 betas[7,0]
   - 259.2758948761588 betas[7,2] - betas[7,3] + tau[21,7] = 0
 tau_computation_21_8: - 15.00272982448498 betas[8,0]
   - 259.2758948761588 betas[8,2] - betas[8,3] + tau[21,8] = 0
 tau_gap_21_0: tau[21,0] - 5e+06 R[21,0] - tau_star_21 <= -0.001
 tau_gap_21_1: tau[21,1] - 5e+06 R[21,1] - tau_star_21 <= -0.001
 tau_gap_21_2: tau[21,2] - 5e+06 R[21,2] - tau_star_21 <= -0.001
 tau_gap_21_3: tau[21,3] - 5e+06 R[21,3] - tau_star_21 <= -0.001
 tau_gap_21_4: tau[21,4] - 5e+06 R[21,4] - tau_star_21 <= -0.001
 tau_gap_21_5: tau[21,5] - 5e+06 R[21,5] - tau_star_21 <= -0.001
 tau_gap_21_6: tau[21,6] - 5e+06 R[21,6] - tau_star_21 <= -0.001
 tau_gap_21_7: tau[21,7] - 5e+06 R[21,7] - tau_star_21 <= -0.001
 tau_gap_21_8: tau[21,8] - 5e+06 R[21,8] - tau_star_21 <= -0.001
 one_R_21: R[21,0] + R[21,1] + R[21,2] + R[21,3] + R[21,4] + R[21,5]
   + R[21,6] + R[21,7] + R[21,8] = 1
 Define_Q_hat_21: - 2602.811304375826 R[21,0] - 2593.393476554932 R[21,1]
   - 2593.408765835199 R[21,2] - 2593.408764633309 R[21,3]
   - 2594.647188093651 R[21,4] - 2594.647187745272 R[21,5]
   - 2597.349924257054 R[21,6] - 2595.807372890313 R[21,7] + Q_hat[21] = 0
 Adjust_Q_hat_21: Q0_var[21] - Q_hat[21] + Q_hat_adjusted[21] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_21: Total_Demand_Up_to_K_minus_1[21]
   - 258.71828521836 R[21,0] - 517.9941803840022 R[21,1]
   - 777.0533132101236 R[21,2] - 1036.329208102474 R[21,3]
   - 1297.069343615687 R[21,4] - 1556.345238491846 R[21,5]
   - 1818.386865494433 R[21,6] - 2077.255582347432 R[21,7] = 0
 Constr_Total_Demand_from_K_to_T_21: Total_Demand_from_k_to_T[21]
   - 2337.089086881389 R[21,0] - 2077.813191715747 R[21,1]
   - 1818.754058889626 R[21,2] - 1559.478163997276 R[21,3]
   - 1298.738028484063 R[21,4] - 1039.462133607904 R[21,5]
   - 777.4205066053171 R[21,6] - 518.5517897523176 R[21,7] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: - Total_Demand_Up_to_K_minus_1[21] + Q0_var[21]
   + Lost_0_aux_21 = 0
 Constr_Q1_plus_left_21: - Left_0[21] + Q1_plus_left[21] - Q1_var[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_1_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: - Total_Demand_from_k_to_T[21]
   + Q1_plus_left[21] + Lost_1_aux_21 = 0
 Profit_Constraint_21: - 600 Sold_0[21] + 600 Lost_0[21] - 600 Sold_1[21]
   + 400 Left_1[21] + 600 Lost_1[21] + profits_vars[21] = 0
 R682: - 17.12700862677528 alphas[0] - 276.633429373568 alphas[2]
   - alphas[3] + f_var[22] = 0
 Q0_upper_bound_22:
   - 2806.409713277834 Fraction_for_second_order_amount[22] + Q0_var[22]
   = 0
 tau_computation_22_0: - 17.12700862677528 betas[0,0]
   - 276.633429373568 betas[0,2] - betas[0,3] + tau[22,0] = 0
 tau_computation_22_1: - 17.12700862677528 betas[1,0]
   - 276.633429373568 betas[1,2] - betas[1,3] + tau[22,1] = 0
 tau_computation_22_2: - 17.12700862677528 betas[2,0]
   - 276.633429373568 betas[2,2] - betas[2,3] + tau[22,2] = 0
 tau_computation_22_3: - 17.12700862677528 betas[3,0]
   - 276.633429373568 betas[3,2] - betas[3,3] + tau[22,3] = 0
 tau_computation_22_4: - 17.12700862677528 betas[4,0]
   - 276.633429373568 betas[4,2] - betas[4,3] + tau[22,4] = 0
 tau_computation_22_5: - 17.12700862677528 betas[5,0]
   - 276.633429373568 betas[5,2] - betas[5,3] + tau[22,5] = 0
 tau_computation_22_6: - 17.12700862677528 betas[6,0]
   - 276.633429373568 betas[6,2] - betas[6,3] + tau[22,6] = 0
 tau_computation_22_7: - 17.12700862677528 betas[7,0]
   - 276.633429373568 betas[7,2] - betas[7,3] + tau[22,7] = 0
 tau_computation_22_8: - 17.12700862677528 betas[8,0]
   - 276.633429373568 betas[8,2] - betas[8,3] + tau[22,8] = 0
 tau_gap_22_0: tau[22,0] - 5e+06 R[22,0] - tau_star_22 <= -0.001
 tau_gap_22_1: tau[22,1] - 5e+06 R[22,1] - tau_star_22 <= -0.001
 tau_gap_22_2: tau[22,2] - 5e+06 R[22,2] - tau_star_22 <= -0.001
 tau_gap_22_3: tau[22,3] - 5e+06 R[22,3] - tau_star_22 <= -0.001
 tau_gap_22_4: tau[22,4] - 5e+06 R[22,4] - tau_star_22 <= -0.001
 tau_gap_22_5: tau[22,5] - 5e+06 R[22,5] - tau_star_22 <= -0.001
 tau_gap_22_6: tau[22,6] - 5e+06 R[22,6] - tau_star_22 <= -0.001
 tau_gap_22_7: tau[22,7] - 5e+06 R[22,7] - tau_star_22 <= -0.001
 tau_gap_22_8: tau[22,8] - 5e+06 R[22,8] - tau_star_22 <= -0.001
 one_R_22: R[22,0] + R[22,1] + R[22,2] + R[22,3] + R[22,4] + R[22,5]
   + R[22,6] + R[22,7] + R[22,8] = 1
 Define_Q_hat_22: - 2754.278387921982 R[22,0] - 2763.967431706367 R[22,1]
   - 2764.034057467158 R[22,2] - 2764.034057335496 R[22,3]
   - 2768.898079116486 R[22,4] - 2768.898079076062 R[22,5]
   - 2766.265581708226 R[22,6] - 2768.786841356316 R[22,7] + Q_hat[22] = 0
 Adjust_Q_hat_22: Q0_var[22] - Q_hat[22] + Q_hat_adjusted[22] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_22: Total_Demand_Up_to_K_minus_1[22]
   - 274.9773369020945 R[22,0] - 551.6107662592763 R[22,1]
   - 827.009177104035 R[22,2] - 1103.642606469471 R[22,3]
   - 1383.933850303963 R[22,4] - 1660.567279677531 R[22,5]
   - 1937.226578415609 R[22,6] - 2215.519981935733 R[22,7] = 0
 Constr_Total_Demand_from_K_to_T_22: Total_Demand_from_k_to_T[22]
   - 2493.809503780774 R[22,0] - 2217.176074423593 R[22,1]
   - 1941.777663578834 R[22,2] - 1665.144234213398 R[22,3]
   - 1384.852990378906 R[22,4] - 1108.219561005338 R[22,5]
   - 831.5602622672595 R[22,6] - 553.2668587471361 R[22,7] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: - Total_Demand_Up_to_K_minus_1[22] + Q0_var[22]
   + Lost_0_aux_22 = 0
 Constr_Q1_plus_left_22: - Left_0[22] + Q1_plus_left[22] - Q1_var[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_1_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: - Total_Demand_from_k_to_T[22]
   + Q1_plus_left[22] + Lost_1_aux_22 = 0
 Profit_Constraint_22: - 600 Sold_0[22] + 600 Lost_0[22] - 600 Sold_1[22]
   + 400 Left_1[22] + 600 Lost_1[22] + profits_vars[22] = 0
 R713: - 15.31776887418079 alphas[0] - 266.3134816324686 alphas[2]
   - alphas[3] + f_var[23] = 0
 Q0_upper_bound_23:
   - 2806.409713277834 Fraction_for_second_order_amount[23] + Q0_var[23]
   = 0
 tau_computation_23_0: - 15.31776887418079 betas[0,0]
   - 266.3134816324686 betas[0,2] - betas[0,3] + tau[23,0] = 0
 tau_computation_23_1: - 15.31776887418079 betas[1,0]
   - 266.3134816324686 betas[1,2] - betas[1,3] + tau[23,1] = 0
 tau_computation_23_2: - 15.31776887418079 betas[2,0]
   - 266.3134816324686 betas[2,2] - betas[2,3] + tau[23,2] = 0
 tau_computation_23_3: - 15.31776887418079 betas[3,0]
   - 266.3134816324686 betas[3,2] - betas[3,3] + tau[23,3] = 0
 tau_computation_23_4: - 15.31776887418079 betas[4,0]
   - 266.3134816324686 betas[4,2] - betas[4,3] + tau[23,4] = 0
 tau_computation_23_5: - 15.31776887418079 betas[5,0]
   - 266.3134816324686 betas[5,2] - betas[5,3] + tau[23,5] = 0
 tau_computation_23_6: - 15.31776887418079 betas[6,0]
   - 266.3134816324686 betas[6,2] - betas[6,3] + tau[23,6] = 0
 tau_computation_23_7: - 15.31776887418079 betas[7,0]
   - 266.3134816324686 betas[7,2] - betas[7,3] + tau[23,7] = 0
 tau_computation_23_8: - 15.31776887418079 betas[8,0]
   - 266.3134816324686 betas[8,2] - betas[8,3] + tau[23,8] = 0
 tau_gap_23_0: tau[23,0] - 5e+06 R[23,0] - tau_star_23 <= -0.001
 tau_gap_23_1: tau[23,1] - 5e+06 R[23,1] - tau_star_23 <= -0.001
 tau_gap_23_2: tau[23,2] - 5e+06 R[23,2] - tau_star_23 <= -0.001
 tau_gap_23_3: tau[23,3] - 5e+06 R[23,3] - tau_star_23 <= -0.001
 tau_gap_23_4: tau[23,4] - 5e+06 R[23,4] - tau_star_23 <= -0.001
 tau_gap_23_5: tau[23,5] - 5e+06 R[23,5] - tau_star_23 <= -0.001
 tau_gap_23_6: tau[23,6] - 5e+06 R[23,6] - tau_star_23 <= -0.001
 tau_gap_23_7: tau[23,7] - 5e+06 R[23,7] - tau_star_23 <= -0.001
 tau_gap_23_8: tau[23,8] - 5e+06 R[23,8] - tau_star_23 <= -0.001
 one_R_23: R[23,0] + R[23,1] + R[23,2] + R[23,3] + R[23,4] + R[23,5]
   + R[23,6] + R[23,7] + R[23,8] = 1
 Define_Q_hat_23: - 2672.050995757802 R[23,0] - 2663.714916031022 R[23,1]
   - 2663.932250238557 R[23,2] - 2663.932249936838 R[23,3]
   - 2665.533956219215 R[23,4] - 2665.533956133889 R[23,5]
   - 2667.576270515478 R[23,6] - 2668.484430475628 R[23,7] + Q_hat[23] = 0
 Adjust_Q_hat_23: Q0_var[23] - Q_hat[23] + Q_hat_adjusted[23] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_23: Total_Demand_Up_to_K_minus_1[23]
   - 266.1507365100119 R[23,0] - 532.464218168307 R[23,1]
   - 797.119485371945 R[23,2] - 1063.432967006938 R[23,3]
   - 1331.962195667689 R[23,4] - 1598.275677300158 R[23,5]
   - 1868.227765246331 R[23,6] - 2135.857466582127 R[23,7] = 0
 Constr_Total_Demand_from_K_to_T_23: Total_Demand_from_k_to_T[23]
   - 2402.333693337052 R[23,0] - 2136.020211678757 R[23,1]
   - 1871.36494447512 R[23,2] - 1605.051462840127 R[23,3]
   - 1336.522234179375 R[23,4] - 1070.208752546907 R[23,5]
   - 800.2566646007335 R[23,6] - 532.6269632649372 R[23,7] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: - Total_Demand_Up_to_K_minus_1[23] + Q0_var[23]
   + Lost_0_aux_23 = 0
 Constr_Q1_plus_left_23: - Left_0[23] + Q1_plus_left[23] - Q1_var[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_1_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: - Total_Demand_from_k_to_T[23]
   + Q1_plus_left[23] + Lost_1_aux_23 = 0
 Profit_Constraint_23: - 600 Sold_0[23] + 600 Lost_0[23] - 600 Sold_1[23]
   + 400 Left_1[23] + 600 Lost_1[23] + profits_vars[23] = 0
 R744: - 16.04126626060742 alphas[0] - 265.8271279946238 alphas[2]
   - alphas[3] + f_var[24] = 0
 Q0_upper_bound_24:
   - 2806.409713277834 Fraction_for_second_order_amount[24] + Q0_var[24]
   = 0
 tau_computation_24_0: - 16.04126626060742 betas[0,0]
   - 265.8271279946238 betas[0,2] - betas[0,3] + tau[24,0] = 0
 tau_computation_24_1: - 16.04126626060742 betas[1,0]
   - 265.8271279946238 betas[1,2] - betas[1,3] + tau[24,1] = 0
 tau_computation_24_2: - 16.04126626060742 betas[2,0]
   - 265.8271279946238 betas[2,2] - betas[2,3] + tau[24,2] = 0
 tau_computation_24_3: - 16.04126626060742 betas[3,0]
   - 265.8271279946238 betas[3,2] - betas[3,3] + tau[24,3] = 0
 tau_computation_24_4: - 16.04126626060742 betas[4,0]
   - 265.8271279946238 betas[4,2] - betas[4,3] + tau[24,4] = 0
 tau_computation_24_5: - 16.04126626060742 betas[5,0]
   - 265.8271279946238 betas[5,2] - betas[5,3] + tau[24,5] = 0
 tau_computation_24_6: - 16.04126626060742 betas[6,0]
   - 265.8271279946238 betas[6,2] - betas[6,3] + tau[24,6] = 0
 tau_computation_24_7: - 16.04126626060742 betas[7,0]
   - 265.8271279946238 betas[7,2] - betas[7,3] + tau[24,7] = 0
 tau_computation_24_8: - 16.04126626060742 betas[8,0]
   - 265.8271279946238 betas[8,2] - betas[8,3] + tau[24,8] = 0
 tau_gap_24_0: tau[24,0] - 5e+06 R[24,0] - tau_star_24 <= -0.001
 tau_gap_24_1: tau[24,1] - 5e+06 R[24,1] - tau_star_24 <= -0.001
 tau_gap_24_2: tau[24,2] - 5e+06 R[24,2] - tau_star_24 <= -0.001
 tau_gap_24_3: tau[24,3] - 5e+06 R[24,3] - tau_star_24 <= -0.001
 tau_gap_24_4: tau[24,4] - 5e+06 R[24,4] - tau_star_24 <= -0.001
 tau_gap_24_5: tau[24,5] - 5e+06 R[24,5] - tau_star_24 <= -0.001
 tau_gap_24_6: tau[24,6] - 5e+06 R[24,6] - tau_star_24 <= -0.001
 tau_gap_24_7: tau[24,7] - 5e+06 R[24,7] - tau_star_24 <= -0.001
 tau_gap_24_8: tau[24,8] - 5e+06 R[24,8] - tau_star_24 <= -0.001
 one_R_24: R[24,0] + R[24,1] + R[24,2] + R[24,3] + R[24,4] + R[24,5]
   + R[24,6] + R[24,7] + R[24,8] = 1
 Define_Q_hat_24: - 2699.134013930948 R[24,0] - 2663.587888797536 R[24,1]
   - 2664.367743098067 R[24,2] - 2664.367744563906 R[24,3]
   - 2668.126869029496 R[24,4] - 2668.126869429162 R[24,5]
   - 2665.68305843049 R[24,6] - 2663.736405774643 R[24,7] + Q_hat[24] = 0
 Adjust_Q_hat_24: Q0_var[24] - Q_hat[24] + Q_hat_adjusted[24] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_24: Total_Demand_Up_to_K_minus_1[24]
   - 269.0579304819254 R[24,0] - 534.8850580013365 R[24,1]
   - 796.1912956489108 R[24,2] - 1062.018423631272 R[24,3]
   - 1333.631584058189 R[24,4] - 1599.458712052813 R[24,5]
   - 1869.098965208442 R[24,6] - 2132.082149341824 R[24,7] = 0
 Constr_Total_Demand_from_K_to_T_24: Total_Demand_from_k_to_T[24]
   - 2394.678474849146 R[24,0] - 2128.851347329735 R[24,1]
   - 1867.545109682161 R[24,2] - 1601.7179816998 R[24,3]
   - 1330.104821272883 R[24,4] - 1064.277693278259 R[24,5]
   - 794.6374401226299 R[24,6] - 531.6542559892476 R[24,7] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: - Total_Demand_Up_to_K_minus_1[24] + Q0_var[24]
   + Lost_0_aux_24 = 0
 Constr_Q1_plus_left_24: - Left_0[24] + Q1_plus_left[24] - Q1_var[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_1_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: - Total_Demand_from_k_to_T[24]
   + Q1_plus_left[24] + Lost_1_aux_24 = 0
 Profit_Constraint_24: - 600 Sold_0[24] + 600 Lost_0[24] - 600 Sold_1[24]
   + 400 Left_1[24] + 600 Lost_1[24] + profits_vars[24] = 0
 R775: - 19.66196969480247 alphas[0] - 272.34384819731 alphas[2]
   - alphas[3] + f_var[25] = 0
 Q0_upper_bound_25:
   - 2806.409713277834 Fraction_for_second_order_amount[25] + Q0_var[25]
   = 0
 tau_computation_25_0: - 19.66196969480247 betas[0,0]
   - 272.34384819731 betas[0,2] - betas[0,3] + tau[25,0] = 0
 tau_computation_25_1: - 19.66196969480247 betas[1,0]
   - 272.34384819731 betas[1,2] - betas[1,3] + tau[25,1] = 0
 tau_computation_25_2: - 19.66196969480247 betas[2,0]
   - 272.34384819731 betas[2,2] - betas[2,3] + tau[25,2] = 0
 tau_computation_25_3: - 19.66196969480247 betas[3,0]
   - 272.34384819731 betas[3,2] - betas[3,3] + tau[25,3] = 0
 tau_computation_25_4: - 19.66196969480247 betas[4,0]
   - 272.34384819731 betas[4,2] - betas[4,3] + tau[25,4] = 0
 tau_computation_25_5: - 19.66196969480247 betas[5,0]
   - 272.34384819731 betas[5,2] - betas[5,3] + tau[25,5] = 0
 tau_computation_25_6: - 19.66196969480247 betas[6,0]
   - 272.34384819731 betas[6,2] - betas[6,3] + tau[25,6] = 0
 tau_computation_25_7: - 19.66196969480247 betas[7,0]
   - 272.34384819731 betas[7,2] - betas[7,3] + tau[25,7] = 0
 tau_computation_25_8: - 19.66196969480247 betas[8,0]
   - 272.34384819731 betas[8,2] - betas[8,3] + tau[25,8] = 0
 tau_gap_25_0: tau[25,0] - 5e+06 R[25,0] - tau_star_25 <= -0.001
 tau_gap_25_1: tau[25,1] - 5e+06 R[25,1] - tau_star_25 <= -0.001
 tau_gap_25_2: tau[25,2] - 5e+06 R[25,2] - tau_star_25 <= -0.001
 tau_gap_25_3: tau[25,3] - 5e+06 R[25,3] - tau_star_25 <= -0.001
 tau_gap_25_4: tau[25,4] - 5e+06 R[25,4] - tau_star_25 <= -0.001
 tau_gap_25_5: tau[25,5] - 5e+06 R[25,5] - tau_star_25 <= -0.001
 tau_gap_25_6: tau[25,6] - 5e+06 R[25,6] - tau_star_25 <= -0.001
 tau_gap_25_7: tau[25,7] - 5e+06 R[25,7] - tau_star_25 <= -0.001
 tau_gap_25_8: tau[25,8] - 5e+06 R[25,8] - tau_star_25 <= -0.001
 one_R_25: R[25,0] + R[25,1] + R[25,2] + R[25,3] + R[25,4] + R[25,5]
   + R[25,6] + R[25,7] + R[25,8] = 1
 Define_Q_hat_25: - 2757.575574641305 R[25,0] - 2727.862113727859 R[25,1]
   - 2727.865149793851 R[25,2] - 2727.865149081532 R[25,3]
   - 2728.993570926258 R[25,4] - 2728.993570751535 R[25,5]
   - 2728.258768319252 R[25,6] - 2728.212410301031 R[25,7] + Q_hat[25] = 0
 Adjust_Q_hat_25: Q0_var[25] - Q_hat[25] + Q_hat_adjusted[25] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_25: Total_Demand_Up_to_K_minus_1[25]
   - 275.3312694399456 R[25,0] - 547.6751178340623 R[25,1]
   - 821.2317389761462 R[25,2] - 1093.575587171597 R[25,3]
   - 1366.616301963466 R[25,4] - 1638.960150160775 R[25,5]
   - 1911.44822528881 R[25,6] - 2183.524713164946 R[25,7] = 0
 Constr_Total_Demand_from_K_to_T_25: Total_Demand_from_k_to_T[25]
   - 2452.88114011962 R[25,0] - 2180.537291725504 R[25,1]
   - 1906.980670583419 R[25,2] - 1634.636822387969 R[25,3]
   - 1361.5961075961 R[25,4] - 1089.25225939879 R[25,5]
   - 816.7641842707559 R[25,6] - 544.6876963946199 R[25,7] = 0
 Constr_Left_0_diff_aux_25: Sold_0[25] - Q0_var[25] + Left_0_aux_25 = 0
 Constr_Lost_0_diff_aux_25: - Total_Demand_Up_to_K_minus_1[25] + Q0_var[25]
   + Lost_0_aux_25 = 0
 Constr_Q1_plus_left_25: - Left_0[25] + Q1_plus_left[25] - Q1_var[25] = 0
 Constr_Left_1_diff_aux_25: Sold_1[25] - Q1_plus_left[25] + Left_1_aux_25
   = 0
 Constr_Lost_1_diff_aux_25: - Total_Demand_from_k_to_T[25]
   + Q1_plus_left[25] + Lost_1_aux_25 = 0
 Profit_Constraint_25: - 600 Sold_0[25] + 600 Lost_0[25] - 600 Sold_1[25]
   + 400 Left_1[25] + 600 Lost_1[25] + profits_vars[25] = 0
 R806: - 16.07699102171619 alphas[0] - 271.6538724550634 alphas[2]
   - alphas[3] + f_var[26] = 0
 Q0_upper_bound_26:
   - 2806.409713277834 Fraction_for_second_order_amount[26] + Q0_var[26]
   = 0
 tau_computation_26_0: - 16.07699102171619 betas[0,0]
   - 271.6538724550634 betas[0,2] - betas[0,3] + tau[26,0] = 0
 tau_computation_26_1: - 16.07699102171619 betas[1,0]
   - 271.6538724550634 betas[1,2] - betas[1,3] + tau[26,1] = 0
 tau_computation_26_2: - 16.07699102171619 betas[2,0]
   - 271.6538724550634 betas[2,2] - betas[2,3] + tau[26,2] = 0
 tau_computation_26_3: - 16.07699102171619 betas[3,0]
   - 271.6538724550634 betas[3,2] - betas[3,3] + tau[26,3] = 0
 tau_computation_26_4: - 16.07699102171619 betas[4,0]
   - 271.6538724550634 betas[4,2] - betas[4,3] + tau[26,4] = 0
 tau_computation_26_5: - 16.07699102171619 betas[5,0]
   - 271.6538724550634 betas[5,2] - betas[5,3] + tau[26,5] = 0
 tau_computation_26_6: - 16.07699102171619 betas[6,0]
   - 271.6538724550634 betas[6,2] - betas[6,3] + tau[26,6] = 0
 tau_computation_26_7: - 16.07699102171619 betas[7,0]
   - 271.6538724550634 betas[7,2] - betas[7,3] + tau[26,7] = 0
 tau_computation_26_8: - 16.07699102171619 betas[8,0]
   - 271.6538724550634 betas[8,2] - betas[8,3] + tau[26,8] = 0
 tau_gap_26_0: tau[26,0] - 5e+06 R[26,0] - tau_star_26 <= -0.001
 tau_gap_26_1: tau[26,1] - 5e+06 R[26,1] - tau_star_26 <= -0.001
 tau_gap_26_2: tau[26,2] - 5e+06 R[26,2] - tau_star_26 <= -0.001
 tau_gap_26_3: tau[26,3] - 5e+06 R[26,3] - tau_star_26 <= -0.001
 tau_gap_26_4: tau[26,4] - 5e+06 R[26,4] - tau_star_26 <= -0.001
 tau_gap_26_5: tau[26,5] - 5e+06 R[26,5] - tau_star_26 <= -0.001
 tau_gap_26_6: tau[26,6] - 5e+06 R[26,6] - tau_star_26 <= -0.001
 tau_gap_26_7: tau[26,7] - 5e+06 R[26,7] - tau_star_26 <= -0.001
 tau_gap_26_8: tau[26,8] - 5e+06 R[26,8] - tau_star_26 <= -0.001
 one_R_26: R[26,0] + R[26,1] + R[26,2] + R[26,3] + R[26,4] + R[26,5]
   + R[26,6] + R[26,7] + R[26,8] = 1
 Define_Q_hat_26: - 2727.156478100355 R[26,0] - 2717.458269468652 R[26,1]
   - 2716.883663699688 R[26,2] - 2716.883663669047 R[26,3]
   - 2717.196223681645 R[26,4] - 2717.196223671071 R[26,5]
   - 2719.476691494242 R[26,6] - 2717.888905699063 R[26,7] + Q_hat[26] = 0
 Adjust_Q_hat_26: Q0_var[26] - Q_hat[26] + Q_hat_adjusted[26] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_26: Total_Demand_Up_to_K_minus_1[26]
   - 272.065968153009 R[26,0] - 543.7198407456647 R[26,1]
   - 819.9144354603511 R[26,2] - 1091.568307923503 R[26,3]
   - 1361.263001755455 R[26,4] - 1632.916874210518 R[26,5]
   - 1903.610531012283 R[26,6] - 2174.581160184512 R[26,7] = 0
 Constr_Total_Demand_from_K_to_T_26: Total_Demand_from_k_to_T[26]
   - 2445.82293694163 R[26,0] - 2174.169064348975 R[26,1]
   - 1897.974469634288 R[26,2] - 1626.320597171136 R[26,3]
   - 1356.625903339185 R[26,4] - 1084.972030884121 R[26,5]
   - 814.2783740823559 R[26,6] - 543.3077449101269 R[26,7] = 0
 Constr_Left_0_diff_aux_26: Sold_0[26] - Q0_var[26] + Left_0_aux_26 = 0
 Constr_Lost_0_diff_aux_26: - Total_Demand_Up_to_K_minus_1[26] + Q0_var[26]
   + Lost_0_aux_26 = 0
 Constr_Q1_plus_left_26: - Left_0[26] + Q1_plus_left[26] - Q1_var[26] = 0
 Constr_Left_1_diff_aux_26: Sold_1[26] - Q1_plus_left[26] + Left_1_aux_26
   = 0
 Constr_Lost_1_diff_aux_26: - Total_Demand_from_k_to_T[26]
   + Q1_plus_left[26] + Lost_1_aux_26 = 0
 Profit_Constraint_26: - 600 Sold_0[26] + 600 Lost_0[26] - 600 Sold_1[26]
   + 400 Left_1[26] + 600 Lost_1[26] + profits_vars[26] = 0
 R837: - 19.29168819317131 alphas[0] - 267.8673439838977 alphas[2]
   - alphas[3] + f_var[27] = 0
 Q0_upper_bound_27:
   - 2806.409713277834 Fraction_for_second_order_amount[27] + Q0_var[27]
   = 0
 tau_computation_27_0: - 19.29168819317131 betas[0,0]
   - 267.8673439838977 betas[0,2] - betas[0,3] + tau[27,0] = 0
 tau_computation_27_1: - 19.29168819317131 betas[1,0]
   - 267.8673439838977 betas[1,2] - betas[1,3] + tau[27,1] = 0
 tau_computation_27_2: - 19.29168819317131 betas[2,0]
   - 267.8673439838977 betas[2,2] - betas[2,3] + tau[27,2] = 0
 tau_computation_27_3: - 19.29168819317131 betas[3,0]
   - 267.8673439838977 betas[3,2] - betas[3,3] + tau[27,3] = 0
 tau_computation_27_4: - 19.29168819317131 betas[4,0]
   - 267.8673439838977 betas[4,2] - betas[4,3] + tau[27,4] = 0
 tau_computation_27_5: - 19.29168819317131 betas[5,0]
   - 267.8673439838977 betas[5,2] - betas[5,3] + tau[27,5] = 0
 tau_computation_27_6: - 19.29168819317131 betas[6,0]
   - 267.8673439838977 betas[6,2] - betas[6,3] + tau[27,6] = 0
 tau_computation_27_7: - 19.29168819317131 betas[7,0]
   - 267.8673439838977 betas[7,2] - betas[7,3] + tau[27,7] = 0
 tau_computation_27_8: - 19.29168819317131 betas[8,0]
   - 267.8673439838977 betas[8,2] - betas[8,3] + tau[27,8] = 0
 tau_gap_27_0: tau[27,0] - 5e+06 R[27,0] - tau_star_27 <= -0.001
 tau_gap_27_1: tau[27,1] - 5e+06 R[27,1] - tau_star_27 <= -0.001
 tau_gap_27_2: tau[27,2] - 5e+06 R[27,2] - tau_star_27 <= -0.001
 tau_gap_27_3: tau[27,3] - 5e+06 R[27,3] - tau_star_27 <= -0.001
 tau_gap_27_4: tau[27,4] - 5e+06 R[27,4] - tau_star_27 <= -0.001
 tau_gap_27_5: tau[27,5] - 5e+06 R[27,5] - tau_star_27 <= -0.001
 tau_gap_27_6: tau[27,6] - 5e+06 R[27,6] - tau_star_27 <= -0.001
 tau_gap_27_7: tau[27,7] - 5e+06 R[27,7] - tau_star_27 <= -0.001
 tau_gap_27_8: tau[27,8] - 5e+06 R[27,8] - tau_star_27 <= -0.001
 one_R_27: R[27,0] + R[27,1] + R[27,2] + R[27,3] + R[27,4] + R[27,5]
   + R[27,6] + R[27,7] + R[27,8] = 1
 Define_Q_hat_27: - 2662.074446773624 R[27,0] - 2675.489493076593 R[27,1]
   - 2676.127381495365 R[27,2] - 2676.127382072988 R[27,3]
   - 2675.214557963218 R[27,4] - 2675.21455812934 R[27,5]
   - 2674.519894960772 R[27,6] - 2673.154226979149 R[27,7] + Q_hat[27] = 0
 Adjust_Q_hat_27: Q0_var[27] - Q_hat[27] + Q_hat_adjusted[27] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_27: Total_Demand_Up_to_K_minus_1[27]
   - 265.0798158732543 R[27,0] - 532.9471595530888 R[27,1]
   - 794.8604174199833 R[27,2] - 1062.727761401181 R[27,3]
   - 1332.666310578314 R[27,4] - 1600.533654562212 R[27,5]
   - 1871.424726068943 R[27,6] - 2137.419538494943 R[27,7] = 0
 Constr_Total_Demand_from_K_to_T_27: Total_Demand_from_k_to_T[27]
   - 2408.074410589485 R[27,0] - 2140.20706690965 R[27,1]
   - 1878.293809042756 R[27,2] - 1610.426465061558 R[27,3]
   - 1340.487915884425 R[27,4] - 1072.620571900527 R[27,5]
   - 801.7295003937965 R[27,6] - 535.7346879677955 R[27,7] = 0
 Constr_Left_0_diff_aux_27: Sold_0[27] - Q0_var[27] + Left_0_aux_27 = 0
 Constr_Lost_0_diff_aux_27: - Total_Demand_Up_to_K_minus_1[27] + Q0_var[27]
   + Lost_0_aux_27 = 0
 Constr_Q1_plus_left_27: - Left_0[27] + Q1_plus_left[27] - Q1_var[27] = 0
 Constr_Left_1_diff_aux_27: Sold_1[27] - Q1_plus_left[27] + Left_1_aux_27
   = 0
 Constr_Lost_1_diff_aux_27: - Total_Demand_from_k_to_T[27]
   + Q1_plus_left[27] + Lost_1_aux_27 = 0
 Profit_Constraint_27: - 600 Sold_0[27] + 600 Lost_0[27] - 600 Sold_1[27]
   + 400 Left_1[27] + 600 Lost_1[27] + profits_vars[27] = 0
 R868: - 19.01446685780667 alphas[0] - 295.7485385157809 alphas[2]
   - alphas[3] + f_var[28] = 0
 Q0_upper_bound_28:
   - 2806.409713277834 Fraction_for_second_order_amount[28] + Q0_var[28]
   = 0
 tau_computation_28_0: - 19.01446685780667 betas[0,0]
   - 295.7485385157809 betas[0,2] - betas[0,3] + tau[28,0] = 0
 tau_computation_28_1: - 19.01446685780667 betas[1,0]
   - 295.7485385157809 betas[1,2] - betas[1,3] + tau[28,1] = 0
 tau_computation_28_2: - 19.01446685780667 betas[2,0]
   - 295.7485385157809 betas[2,2] - betas[2,3] + tau[28,2] = 0
 tau_computation_28_3: - 19.01446685780667 betas[3,0]
   - 295.7485385157809 betas[3,2] - betas[3,3] + tau[28,3] = 0
 tau_computation_28_4: - 19.01446685780667 betas[4,0]
   - 295.7485385157809 betas[4,2] - betas[4,3] + tau[28,4] = 0
 tau_computation_28_5: - 19.01446685780667 betas[5,0]
   - 295.7485385157809 betas[5,2] - betas[5,3] + tau[28,5] = 0
 tau_computation_28_6: - 19.01446685780667 betas[6,0]
   - 295.7485385157809 betas[6,2] - betas[6,3] + tau[28,6] = 0
 tau_computation_28_7: - 19.01446685780667 betas[7,0]
   - 295.7485385157809 betas[7,2] - betas[7,3] + tau[28,7] = 0
 tau_computation_28_8: - 19.01446685780667 betas[8,0]
   - 295.7485385157809 betas[8,2] - betas[8,3] + tau[28,8] = 0
 tau_gap_28_0: tau[28,0] - 5e+06 R[28,0] - tau_star_28 <= -0.001
 tau_gap_28_1: tau[28,1] - 5e+06 R[28,1] - tau_star_28 <= -0.001
 tau_gap_28_2: tau[28,2] - 5e+06 R[28,2] - tau_star_28 <= -0.001
 tau_gap_28_3: tau[28,3] - 5e+06 R[28,3] - tau_star_28 <= -0.001
 tau_gap_28_4: tau[28,4] - 5e+06 R[28,4] - tau_star_28 <= -0.001
 tau_gap_28_5: tau[28,5] - 5e+06 R[28,5] - tau_star_28 <= -0.001
 tau_gap_28_6: tau[28,6] - 5e+06 R[28,6] - tau_star_28 <= -0.001
 tau_gap_28_7: tau[28,7] - 5e+06 R[28,7] - tau_star_28 <= -0.001
 tau_gap_28_8: tau[28,8] - 5e+06 R[28,8] - tau_star_28 <= -0.001
 one_R_28: R[28,0] + R[28,1] + R[28,2] + R[28,3] + R[28,4] + R[28,5]
   + R[28,6] + R[28,7] + R[28,8] = 1
 Define_Q_hat_28: - 2974.938772363464 R[28,0] - 2959.811455655876 R[28,1]
   - 2959.902583845289 R[28,2] - 2959.902583715707 R[28,3]
   - 2961.378828629894 R[28,4] - 2961.378828640361 R[28,5]
   - 2962.387404874429 R[28,6] - 2960.192472159272 R[28,7] + Q_hat[28] = 0
 Adjust_Q_hat_28: Q0_var[28] - Q_hat[28] + Q_hat_adjusted[28] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_28: Total_Demand_Up_to_K_minus_1[28]
   - 298.6638601465559 R[28,0] - 594.412398675275 R[28,1]
   - 890.4406007719128 R[28,2] - 1186.189139268442 R[28,3]
   - 1482.345488224567 R[28,4] - 1778.094026740348 R[28,5]
   - 2076.543034833912 R[28,6] - 2368.695394514824 R[28,7] = 0
 Constr_Total_Demand_from_K_to_T_28: Total_Demand_from_k_to_T[28]
   - 2661.52861139983 R[28,0] - 2365.780072871111 R[28,1]
   - 2069.751870774473 R[28,2] - 1774.003332277944 R[28,3]
   - 1477.846983321818 R[28,4] - 1182.098444806038 R[28,5]
   - 883.6494367124737 R[28,6] - 591.4970770315618 R[28,7] = 0
 Constr_Left_0_diff_aux_28: Sold_0[28] - Q0_var[28] + Left_0_aux_28 = 0
 Constr_Lost_0_diff_aux_28: - Total_Demand_Up_to_K_minus_1[28] + Q0_var[28]
   + Lost_0_aux_28 = 0
 Constr_Q1_plus_left_28: - Left_0[28] + Q1_plus_left[28] - Q1_var[28] = 0
 Constr_Left_1_diff_aux_28: Sold_1[28] - Q1_plus_left[28] + Left_1_aux_28
   = 0
 Constr_Lost_1_diff_aux_28: - Total_Demand_from_k_to_T[28]
   + Q1_plus_left[28] + Lost_1_aux_28 = 0
 Profit_Constraint_28: - 600 Sold_0[28] + 600 Lost_0[28] - 600 Sold_1[28]
   + 400 Left_1[28] + 600 Lost_1[28] + profits_vars[28] = 0
 R899: - 15.79573118471121 alphas[0] - 286.5872092716446 alphas[2]
   - alphas[3] + f_var[29] = 0
 Q0_upper_bound_29:
   - 2806.409713277834 Fraction_for_second_order_amount[29] + Q0_var[29]
   = 0
 tau_computation_29_0: - 15.79573118471121 betas[0,0]
   - 286.5872092716446 betas[0,2] - betas[0,3] + tau[29,0] = 0
 tau_computation_29_1: - 15.79573118471121 betas[1,0]
   - 286.5872092716446 betas[1,2] - betas[1,3] + tau[29,1] = 0
 tau_computation_29_2: - 15.79573118471121 betas[2,0]
   - 286.5872092716446 betas[2,2] - betas[2,3] + tau[29,2] = 0
 tau_computation_29_3: - 15.79573118471121 betas[3,0]
   - 286.5872092716446 betas[3,2] - betas[3,3] + tau[29,3] = 0
 tau_computation_29_4: - 15.79573118471121 betas[4,0]
   - 286.5872092716446 betas[4,2] - betas[4,3] + tau[29,4] = 0
 tau_computation_29_5: - 15.79573118471121 betas[5,0]
   - 286.5872092716446 betas[5,2] - betas[5,3] + tau[29,5] = 0
 tau_computation_29_6: - 15.79573118471121 betas[6,0]
   - 286.5872092716446 betas[6,2] - betas[6,3] + tau[29,6] = 0
 tau_computation_29_7: - 15.79573118471121 betas[7,0]
   - 286.5872092716446 betas[7,2] - betas[7,3] + tau[29,7] = 0
 tau_computation_29_8: - 15.79573118471121 betas[8,0]
   - 286.5872092716446 betas[8,2] - betas[8,3] + tau[29,8] = 0
 tau_gap_29_0: tau[29,0] - 5e+06 R[29,0] - tau_star_29 <= -0.001
 tau_gap_29_1: tau[29,1] - 5e+06 R[29,1] - tau_star_29 <= -0.001
 tau_gap_29_2: tau[29,2] - 5e+06 R[29,2] - tau_star_29 <= -0.001
 tau_gap_29_3: tau[29,3] - 5e+06 R[29,3] - tau_star_29 <= -0.001
 tau_gap_29_4: tau[29,4] - 5e+06 R[29,4] - tau_star_29 <= -0.001
 tau_gap_29_5: tau[29,5] - 5e+06 R[29,5] - tau_star_29 <= -0.001
 tau_gap_29_6: tau[29,6] - 5e+06 R[29,6] - tau_star_29 <= -0.001
 tau_gap_29_7: tau[29,7] - 5e+06 R[29,7] - tau_star_29 <= -0.001
 tau_gap_29_8: tau[29,8] - 5e+06 R[29,8] - tau_star_29 <= -0.001
 one_R_29: R[29,0] + R[29,1] + R[29,2] + R[29,3] + R[29,4] + R[29,5]
   + R[29,6] + R[29,7] + R[29,8] = 1
 Define_Q_hat_29: - 2875.494624005365 R[29,0] - 2866.886391228169 R[29,1]
   - 2867.084034789286 R[29,2] - 2867.084034739047 R[29,3]
   - 2864.760073515569 R[29,4] - 2864.760073520693 R[29,5]
   - 2863.475071542903 R[29,6] - 2863.48619789294 R[29,7] + Q_hat[29] = 0
 Adjust_Q_hat_29: Q0_var[29] - Q_hat[29] + Q_hat_adjusted[29] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_29: Total_Demand_Up_to_K_minus_1[29]
   - 287.9891477935805 R[29,0] - 574.576357045528 R[29,1]
   - 860.0995993682097 R[29,2] - 1146.686808642386 R[29,3]
   - 1431.769774543382 R[29,4] - 1718.356983815027 R[29,5]
   - 2005.013391805547 R[29,6] - 2290.311778728196 R[29,7] = 0
 Constr_Total_Demand_from_K_to_T_29: Total_Demand_from_k_to_T[29]
   - 2575.497049477905 R[29,0] - 2288.909840225958 R[29,1]
   - 2003.386597903276 R[29,2] - 1716.799388629099 R[29,3]
   - 1431.716422728103 R[29,4] - 1145.129213456459 R[29,5]
   - 858.4728054659387 R[29,6] - 573.1744185432892 R[29,7] = 0
 Constr_Left_0_diff_aux_29: Sold_0[29] - Q0_var[29] + Left_0_aux_29 = 0
 Constr_Lost_0_diff_aux_29: - Total_Demand_Up_to_K_minus_1[29] + Q0_var[29]
   + Lost_0_aux_29 = 0
 Constr_Q1_plus_left_29: - Left_0[29] + Q1_plus_left[29] - Q1_var[29] = 0
 Constr_Left_1_diff_aux_29: Sold_1[29] - Q1_plus_left[29] + Left_1_aux_29
   = 0
 Constr_Lost_1_diff_aux_29: - Total_Demand_from_k_to_T[29]
   + Q1_plus_left[29] + Lost_1_aux_29 = 0
 Profit_Constraint_29: - 600 Sold_0[29] + 600 Lost_0[29] - 600 Sold_1[29]
   + 400 Left_1[29] + 600 Lost_1[29] + profits_vars[29] = 0
 R930: - 18.02855978635139 alphas[0] - 286.3773495665765 alphas[2]
   - alphas[3] + f_var[30] = 0
 Q0_upper_bound_30:
   - 2806.409713277834 Fraction_for_second_order_amount[30] + Q0_var[30]
   = 0
 tau_computation_30_0: - 18.02855978635139 betas[0,0]
   - 286.3773495665765 betas[0,2] - betas[0,3] + tau[30,0] = 0
 tau_computation_30_1: - 18.02855978635139 betas[1,0]
   - 286.3773495665765 betas[1,2] - betas[1,3] + tau[30,1] = 0
 tau_computation_30_2: - 18.02855978635139 betas[2,0]
   - 286.3773495665765 betas[2,2] - betas[2,3] + tau[30,2] = 0
 tau_computation_30_3: - 18.02855978635139 betas[3,0]
   - 286.3773495665765 betas[3,2] - betas[3,3] + tau[30,3] = 0
 tau_computation_30_4: - 18.02855978635139 betas[4,0]
   - 286.3773495665765 betas[4,2] - betas[4,3] + tau[30,4] = 0
 tau_computation_30_5: - 18.02855978635139 betas[5,0]
   - 286.3773495665765 betas[5,2] - betas[5,3] + tau[30,5] = 0
 tau_computation_30_6: - 18.02855978635139 betas[6,0]
   - 286.3773495665765 betas[6,2] - betas[6,3] + tau[30,6] = 0
 tau_computation_30_7: - 18.02855978635139 betas[7,0]
   - 286.3773495665765 betas[7,2] - betas[7,3] + tau[30,7] = 0
 tau_computation_30_8: - 18.02855978635139 betas[8,0]
   - 286.3773495665765 betas[8,2] - betas[8,3] + tau[30,8] = 0
 tau_gap_30_0: tau[30,0] - 5e+06 R[30,0] - tau_star_30 <= -0.001
 tau_gap_30_1: tau[30,1] - 5e+06 R[30,1] - tau_star_30 <= -0.001
 tau_gap_30_2: tau[30,2] - 5e+06 R[30,2] - tau_star_30 <= -0.001
 tau_gap_30_3: tau[30,3] - 5e+06 R[30,3] - tau_star_30 <= -0.001
 tau_gap_30_4: tau[30,4] - 5e+06 R[30,4] - tau_star_30 <= -0.001
 tau_gap_30_5: tau[30,5] - 5e+06 R[30,5] - tau_star_30 <= -0.001
 tau_gap_30_6: tau[30,6] - 5e+06 R[30,6] - tau_star_30 <= -0.001
 tau_gap_30_7: tau[30,7] - 5e+06 R[30,7] - tau_star_30 <= -0.001
 tau_gap_30_8: tau[30,8] - 5e+06 R[30,8] - tau_star_30 <= -0.001
 one_R_30: R[30,0] + R[30,1] + R[30,2] + R[30,3] + R[30,4] + R[30,5]
   + R[30,6] + R[30,7] + R[30,8] = 1
 Define_Q_hat_30: - 2864.766810489657 R[30,0] - 2863.502852431518 R[30,1]
   - 2863.821304895686 R[30,2] - 2863.821304668106 R[30,3]
   - 2862.334757826857 R[30,4] - 2862.33475777926 R[30,5]
   - 2859.921607836654 R[30,6] - 2861.313953217202 R[30,7] + Q_hat[30] = 0
 Adjust_Q_hat_30: Q0_var[30] - Q_hat[30] + Q_hat_adjusted[30] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_30: Total_Demand_Up_to_K_minus_1[30]
   - 286.8375835703729 R[30,0] - 573.2149331210412 R[30,1]
   - 857.2231558968792 R[30,2] - 1143.60050546509 R[30,3]
   - 1429.58408023593 R[30,4] - 1715.961429802506 R[30,5]
   - 2002.384639496945 R[30,6] - 2288.55925342404 R[30,7] = 0
 Constr_Total_Demand_from_K_to_T_30: Total_Demand_from_k_to_T[30]
   - 2574.47636898682 R[30,0] - 2288.099019436151 R[30,1]
   - 2004.090796660313 R[30,2] - 1717.713447092102 R[30,3]
   - 1431.729872321263 R[30,4] - 1145.352522754686 R[30,5]
   - 858.9293130602473 R[30,6] - 572.7546991331529 R[30,7] = 0
 Constr_Left_0_diff_aux_30: Sold_0[30] - Q0_var[30] + Left_0_aux_30 = 0
 Constr_Lost_0_diff_aux_30: - Total_Demand_Up_to_K_minus_1[30] + Q0_var[30]
   + Lost_0_aux_30 = 0
 Constr_Q1_plus_left_30: - Left_0[30] + Q1_plus_left[30] - Q1_var[30] = 0
 Constr_Left_1_diff_aux_30: Sold_1[30] - Q1_plus_left[30] + Left_1_aux_30
   = 0
 Constr_Lost_1_diff_aux_30: - Total_Demand_from_k_to_T[30]
   + Q1_plus_left[30] + Lost_1_aux_30 = 0
 Profit_Constraint_30: - 600 Sold_0[30] + 600 Lost_0[30] - 600 Sold_1[30]
   + 400 Left_1[30] + 600 Lost_1[30] + profits_vars[30] = 0
 R961: - 15.57830935952507 alphas[0] - 264.4956724795978 alphas[2]
   - alphas[3] + f_var[31] = 0
 Q0_upper_bound_31:
   - 2806.409713277834 Fraction_for_second_order_amount[31] + Q0_var[31]
   = 0
 tau_computation_31_0: - 15.57830935952507 betas[0,0]
   - 264.4956724795978 betas[0,2] - betas[0,3] + tau[31,0] = 0
 tau_computation_31_1: - 15.57830935952507 betas[1,0]
   - 264.4956724795978 betas[1,2] - betas[1,3] + tau[31,1] = 0
 tau_computation_31_2: - 15.57830935952507 betas[2,0]
   - 264.4956724795978 betas[2,2] - betas[2,3] + tau[31,2] = 0
 tau_computation_31_3: - 15.57830935952507 betas[3,0]
   - 264.4956724795978 betas[3,2] - betas[3,3] + tau[31,3] = 0
 tau_computation_31_4: - 15.57830935952507 betas[4,0]
   - 264.4956724795978 betas[4,2] - betas[4,3] + tau[31,4] = 0
 tau_computation_31_5: - 15.57830935952507 betas[5,0]
   - 264.4956724795978 betas[5,2] - betas[5,3] + tau[31,5] = 0
 tau_computation_31_6: - 15.57830935952507 betas[6,0]
   - 264.4956724795978 betas[6,2] - betas[6,3] + tau[31,6] = 0
 tau_computation_31_7: - 15.57830935952507 betas[7,0]
   - 264.4956724795978 betas[7,2] - betas[7,3] + tau[31,7] = 0
 tau_computation_31_8: - 15.57830935952507 betas[8,0]
   - 264.4956724795978 betas[8,2] - betas[8,3] + tau[31,8] = 0
 tau_gap_31_0: tau[31,0] - 5e+06 R[31,0] - tau_star_31 <= -0.001
 tau_gap_31_1: tau[31,1] - 5e+06 R[31,1] - tau_star_31 <= -0.001
 tau_gap_31_2: tau[31,2] - 5e+06 R[31,2] - tau_star_31 <= -0.001
 tau_gap_31_3: tau[31,3] - 5e+06 R[31,3] - tau_star_31 <= -0.001
 tau_gap_31_4: tau[31,4] - 5e+06 R[31,4] - tau_star_31 <= -0.001
 tau_gap_31_5: tau[31,5] - 5e+06 R[31,5] - tau_star_31 <= -0.001
 tau_gap_31_6: tau[31,6] - 5e+06 R[31,6] - tau_star_31 <= -0.001
 tau_gap_31_7: tau[31,7] - 5e+06 R[31,7] - tau_star_31 <= -0.001
 tau_gap_31_8: tau[31,8] - 5e+06 R[31,8] - tau_star_31 <= -0.001
 one_R_31: R[31,0] + R[31,1] + R[31,2] + R[31,3] + R[31,4] + R[31,5]
   + R[31,6] + R[31,7] + R[31,8] = 1
 Define_Q_hat_31: - 2648.771474464057 R[31,0] - 2644.749680978929 R[31,1]
   - 2644.658452609688 R[31,2] - 2644.658452230995 R[31,3]
   - 2640.105453617743 R[31,4] - 2640.105453563547 R[31,5]
   - 2636.743201788014 R[31,6] - 2634.601748425627 R[31,7] + Q_hat[31] = 0
 Adjust_Q_hat_31: Q0_var[31] - Q_hat[31] + Q_hat_adjusted[31] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_31: Total_Demand_Up_to_K_minus_1[31]
   - 263.6518243303747 R[31,0] - 528.1474969065664 R[31,1]
   - 793.0646514778495 R[31,2] - 1057.560323955986 R[31,3]
   - 1318.773497408745 R[31,4] - 1583.269169888343 R[31,5]
   - 1843.283589249149 R[31,6] - 2105.610402768063 R[31,7] = 0
 Constr_Total_Demand_from_K_to_T_31: Total_Demand_from_k_to_T[31]
   - 2370.949923396884 R[31,0] - 2106.454250820692 R[31,1]
   - 1841.537096249409 R[31,2] - 1577.041423771272 R[31,3]
   - 1315.828250318513 R[31,4] - 1051.332577838916 R[31,5]
   - 791.3181584781098 R[31,6] - 528.9913449591955 R[31,7] = 0
 Constr_Left_0_diff_aux_31: Sold_0[31] - Q0_var[31] + Left_0_aux_31 = 0
 Constr_Lost_0_diff_aux_31: - Total_Demand_Up_to_K_minus_1[31] + Q0_var[31]
   + Lost_0_aux_31 = 0
 Constr_Q1_plus_left_31: - Left_0[31] + Q1_plus_left[31] - Q1_var[31] = 0
 Constr_Left_1_diff_aux_31: Sold_1[31] - Q1_plus_left[31] + Left_1_aux_31
   = 0
 Constr_Lost_1_diff_aux_31: - Total_Demand_from_k_to_T[31]
   + Q1_plus_left[31] + Lost_1_aux_31 = 0
 Profit_Constraint_31: - 600 Sold_0[31] + 600 Lost_0[31] - 600 Sold_1[31]
   + 400 Left_1[31] + 600 Lost_1[31] + profits_vars[31] = 0
 R992: - 18.63944079184756 alphas[0] - 278.885471215842 alphas[2]
   - alphas[3] + f_var[32] = 0
 Q0_upper_bound_32:
   - 2806.409713277834 Fraction_for_second_order_amount[32] + Q0_var[32]
   = 0
 tau_computation_32_0: - 18.63944079184756 betas[0,0]
   - 278.885471215842 betas[0,2] - betas[0,3] + tau[32,0] = 0
 tau_computation_32_1: - 18.63944079184756 betas[1,0]
   - 278.885471215842 betas[1,2] - betas[1,3] + tau[32,1] = 0
 tau_computation_32_2: - 18.63944079184756 betas[2,0]
   - 278.885471215842 betas[2,2] - betas[2,3] + tau[32,2] = 0
 tau_computation_32_3: - 18.63944079184756 betas[3,0]
   - 278.885471215842 betas[3,2] - betas[3,3] + tau[32,3] = 0
 tau_computation_32_4: - 18.63944079184756 betas[4,0]
   - 278.885471215842 betas[4,2] - betas[4,3] + tau[32,4] = 0
 tau_computation_32_5: - 18.63944079184756 betas[5,0]
   - 278.885471215842 betas[5,2] - betas[5,3] + tau[32,5] = 0
 tau_computation_32_6: - 18.63944079184756 betas[6,0]
   - 278.885471215842 betas[6,2] - betas[6,3] + tau[32,6] = 0
 tau_computation_32_7: - 18.63944079184756 betas[7,0]
   - 278.885471215842 betas[7,2] - betas[7,3] + tau[32,7] = 0
 tau_computation_32_8: - 18.63944079184756 betas[8,0]
   - 278.885471215842 betas[8,2] - betas[8,3] + tau[32,8] = 0
 tau_gap_32_0: tau[32,0] - 5e+06 R[32,0] - tau_star_32 <= -0.001
 tau_gap_32_1: tau[32,1] - 5e+06 R[32,1] - tau_star_32 <= -0.001
 tau_gap_32_2: tau[32,2] - 5e+06 R[32,2] - tau_star_32 <= -0.001
 tau_gap_32_3: tau[32,3] - 5e+06 R[32,3] - tau_star_32 <= -0.001
 tau_gap_32_4: tau[32,4] - 5e+06 R[32,4] - tau_star_32 <= -0.001
 tau_gap_32_5: tau[32,5] - 5e+06 R[32,5] - tau_star_32 <= -0.001
 tau_gap_32_6: tau[32,6] - 5e+06 R[32,6] - tau_star_32 <= -0.001
 tau_gap_32_7: tau[32,7] - 5e+06 R[32,7] - tau_star_32 <= -0.001
 tau_gap_32_8: tau[32,8] - 5e+06 R[32,8] - tau_star_32 <= -0.001
 one_R_32: R[32,0] + R[32,1] + R[32,2] + R[32,3] + R[32,4] + R[32,5]
   + R[32,6] + R[32,7] + R[32,8] = 1
 Define_Q_hat_32: - 2782.831618910775 R[32,0] - 2787.420366974011 R[32,1]
   - 2787.999292141858 R[32,2] - 2787.99929214348 R[32,3]
   - 2783.562441393034 R[32,4] - 2783.562441389855 R[32,5]
   - 2784.421618882935 R[32,6] - 2782.68079923055 R[32,7] + Q_hat[32] = 0
 Adjust_Q_hat_32: Q0_var[32] - Q_hat[32] + Q_hat_adjusted[32] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_32: Total_Demand_Up_to_K_minus_1[32]
   - 278.0423490982455 R[32,0] - 556.9278201927457 R[32,1]
   - 831.0015109700767 R[32,2] - 1109.886982201702 R[32,3]
   - 1387.593001388228 R[32,4] - 1666.47847260407 R[32,5]
   - 1948.584626670957 R[32,6] - 2224.90985621343 R[32,7] = 0
 Constr_Total_Demand_from_K_to_T_32: Total_Demand_from_k_to_T[32]
   - 2504.638449546868 R[32,0] - 2225.752978452369 R[32,1]
   - 1951.679287675038 R[32,2] - 1672.793816443412 R[32,3]
   - 1395.087797256886 R[32,4] - 1116.202326041044 R[32,5]
   - 834.096171974157 R[32,6] - 557.770942431684 R[32,7] = 0
 Constr_Left_0_diff_aux_32: Sold_0[32] - Q0_var[32] + Left_0_aux_32 = 0
 Constr_Lost_0_diff_aux_32: - Total_Demand_Up_to_K_minus_1[32] + Q0_var[32]
   + Lost_0_aux_32 = 0
 Constr_Q1_plus_left_32: - Left_0[32] + Q1_plus_left[32] - Q1_var[32] = 0
 Constr_Left_1_diff_aux_32: Sold_1[32] - Q1_plus_left[32] + Left_1_aux_32
   = 0
 Constr_Lost_1_diff_aux_32: - Total_Demand_from_k_to_T[32]
   + Q1_plus_left[32] + Lost_1_aux_32 = 0
 Profit_Constraint_32: - 600 Sold_0[32] + 600 Lost_0[32] - 600 Sold_1[32]
   + 400 Left_1[32] + 600 Lost_1[32] + profits_vars[32] = 0
 R1023: - 18.18731138686103 alphas[0] - 288.9589716650917 alphas[2]
   - alphas[3] + f_var[33] = 0
 Q0_upper_bound_33:
   - 2806.409713277834 Fraction_for_second_order_amount[33] + Q0_var[33]
   = 0
 tau_computation_33_0: - 18.18731138686103 betas[0,0]
   - 288.9589716650917 betas[0,2] - betas[0,3] + tau[33,0] = 0
 tau_computation_33_1: - 18.18731138686103 betas[1,0]
   - 288.9589716650917 betas[1,2] - betas[1,3] + tau[33,1] = 0
 tau_computation_33_2: - 18.18731138686103 betas[2,0]
   - 288.9589716650917 betas[2,2] - betas[2,3] + tau[33,2] = 0
 tau_computation_33_3: - 18.18731138686103 betas[3,0]
   - 288.9589716650917 betas[3,2] - betas[3,3] + tau[33,3] = 0
 tau_computation_33_4: - 18.18731138686103 betas[4,0]
   - 288.9589716650917 betas[4,2] - betas[4,3] + tau[33,4] = 0
 tau_computation_33_5: - 18.18731138686103 betas[5,0]
   - 288.9589716650917 betas[5,2] - betas[5,3] + tau[33,5] = 0
 tau_computation_33_6: - 18.18731138686103 betas[6,0]
   - 288.9589716650917 betas[6,2] - betas[6,3] + tau[33,6] = 0
 tau_computation_33_7: - 18.18731138686103 betas[7,0]
   - 288.9589716650917 betas[7,2] - betas[7,3] + tau[33,7] = 0
 tau_computation_33_8: - 18.18731138686103 betas[8,0]
   - 288.9589716650917 betas[8,2] - betas[8,3] + tau[33,8] = 0
 tau_gap_33_0: tau[33,0] - 5e+06 R[33,0] - tau_star_33 <= -0.001
 tau_gap_33_1: tau[33,1] - 5e+06 R[33,1] - tau_star_33 <= -0.001
 tau_gap_33_2: tau[33,2] - 5e+06 R[33,2] - tau_star_33 <= -0.001
 tau_gap_33_3: tau[33,3] - 5e+06 R[33,3] - tau_star_33 <= -0.001
 tau_gap_33_4: tau[33,4] - 5e+06 R[33,4] - tau_star_33 <= -0.001
 tau_gap_33_5: tau[33,5] - 5e+06 R[33,5] - tau_star_33 <= -0.001
 tau_gap_33_6: tau[33,6] - 5e+06 R[33,6] - tau_star_33 <= -0.001
 tau_gap_33_7: tau[33,7] - 5e+06 R[33,7] - tau_star_33 <= -0.001
 tau_gap_33_8: tau[33,8] - 5e+06 R[33,8] - tau_star_33 <= -0.001
 one_R_33: R[33,0] + R[33,1] + R[33,2] + R[33,3] + R[33,4] + R[33,5]
   + R[33,6] + R[33,7] + R[33,8] = 1
 Define_Q_hat_33: - 2908.400381797105 R[33,0] - 2892.00707698447 R[33,1]
   - 2892.285847602133 R[33,2] - 2892.285847637912 R[33,3]
   - 2890.289718221302 R[33,4] - 2890.289718258484 R[33,5]
   - 2890.57895011746 R[33,6] - 2889.876027727603 R[33,7] + Q_hat[33] = 0
 Adjust_Q_hat_33: Q0_var[33] - Q_hat[33] + Q_hat_adjusted[33] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_33: Total_Demand_Up_to_K_minus_1[33]
   - 291.5213767379988 R[33,0] - 580.4803483546051 R[33,1]
   - 868.2157983569334 R[33,2] - 1157.174770019708 R[33,3]
   - 1445.036119499216 R[33,4] - 1733.995091164308 R[33,5]
   - 2024.967336806668 R[33,6] - 2311.958083812573 R[33,7] = 0
 Constr_Total_Demand_from_K_to_T_33: Total_Demand_from_k_to_T[33]
   - 2598.354650404758 R[33,0] - 2309.395678788152 R[33,1]
   - 2021.660228785823 R[33,2] - 1732.701257123049 R[33,3]
   - 1444.839907643541 R[33,4] - 1155.880935978449 R[33,5]
   - 864.9086903360887 R[33,6] - 577.9179433301834 R[33,7] = 0
 Constr_Left_0_diff_aux_33: Sold_0[33] - Q0_var[33] + Left_0_aux_33 = 0
 Constr_Lost_0_diff_aux_33: - Total_Demand_Up_to_K_minus_1[33] + Q0_var[33]
   + Lost_0_aux_33 = 0
 Constr_Q1_plus_left_33: - Left_0[33] + Q1_plus_left[33] - Q1_var[33] = 0
 Constr_Left_1_diff_aux_33: Sold_1[33] - Q1_plus_left[33] + Left_1_aux_33
   = 0
 Constr_Lost_1_diff_aux_33: - Total_Demand_from_k_to_T[33]
   + Q1_plus_left[33] + Lost_1_aux_33 = 0
 Profit_Constraint_33: - 600 Sold_0[33] + 600 Lost_0[33] - 600 Sold_1[33]
   + 400 Left_1[33] + 600 Lost_1[33] + profits_vars[33] = 0
 R1054: - 19.0596928084551 alphas[0] - 289.7795184271607 alphas[2]
   - alphas[3] + f_var[34] = 0
 Q0_upper_bound_34:
   - 2806.409713277834 Fraction_for_second_order_amount[34] + Q0_var[34]
   = 0
 tau_computation_34_0: - 19.0596928084551 betas[0,0]
   - 289.7795184271607 betas[0,2] - betas[0,3] + tau[34,0] = 0
 tau_computation_34_1: - 19.0596928084551 betas[1,0]
   - 289.7795184271607 betas[1,2] - betas[1,3] + tau[34,1] = 0
 tau_computation_34_2: - 19.0596928084551 betas[2,0]
   - 289.7795184271607 betas[2,2] - betas[2,3] + tau[34,2] = 0
 tau_computation_34_3: - 19.0596928084551 betas[3,0]
   - 289.7795184271607 betas[3,2] - betas[3,3] + tau[34,3] = 0
 tau_computation_34_4: - 19.0596928084551 betas[4,0]
   - 289.7795184271607 betas[4,2] - betas[4,3] + tau[34,4] = 0
 tau_computation_34_5: - 19.0596928084551 betas[5,0]
   - 289.7795184271607 betas[5,2] - betas[5,3] + tau[34,5] = 0
 tau_computation_34_6: - 19.0596928084551 betas[6,0]
   - 289.7795184271607 betas[6,2] - betas[6,3] + tau[34,6] = 0
 tau_computation_34_7: - 19.0596928084551 betas[7,0]
   - 289.7795184271607 betas[7,2] - betas[7,3] + tau[34,7] = 0
 tau_computation_34_8: - 19.0596928084551 betas[8,0]
   - 289.7795184271607 betas[8,2] - betas[8,3] + tau[34,8] = 0
 tau_gap_34_0: tau[34,0] - 5e+06 R[34,0] - tau_star_34 <= -0.001
 tau_gap_34_1: tau[34,1] - 5e+06 R[34,1] - tau_star_34 <= -0.001
 tau_gap_34_2: tau[34,2] - 5e+06 R[34,2] - tau_star_34 <= -0.001
 tau_gap_34_3: tau[34,3] - 5e+06 R[34,3] - tau_star_34 <= -0.001
 tau_gap_34_4: tau[34,4] - 5e+06 R[34,4] - tau_star_34 <= -0.001
 tau_gap_34_5: tau[34,5] - 5e+06 R[34,5] - tau_star_34 <= -0.001
 tau_gap_34_6: tau[34,6] - 5e+06 R[34,6] - tau_star_34 <= -0.001
 tau_gap_34_7: tau[34,7] - 5e+06 R[34,7] - tau_star_34 <= -0.001
 tau_gap_34_8: tau[34,8] - 5e+06 R[34,8] - tau_star_34 <= -0.001
 one_R_34: R[34,0] + R[34,1] + R[34,2] + R[34,3] + R[34,4] + R[34,5]
   + R[34,6] + R[34,7] + R[34,8] = 1
 Define_Q_hat_34: - 2899.809767282738 R[34,0] - 2897.731470707145 R[34,1]
   - 2897.787527956969 R[34,2] - 2897.787527929134 R[34,3]
   - 2891.475196165173 R[34,4] - 2891.475196210388 R[34,5]
   - 2887.997263640038 R[34,6] - 2888.80159106743 R[34,7] + Q_hat[34] = 0
 Adjust_Q_hat_34: Q0_var[34] - Q_hat[34] + Q_hat_adjusted[34] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_34: Total_Demand_Up_to_K_minus_1[34]
   - 290.5992275679346 R[34,0] - 580.3787459872724 R[34,1]
   - 869.8966569093716 R[34,2] - 1159.676175335375 R[34,3]
   - 1444.453441701986 R[34,4] - 1734.232960129146 R[34,5]
   - 2020.33568585418 R[34,6] - 2309.242553599348 R[34,7] = 0
 Constr_Total_Demand_from_K_to_T_34: Total_Demand_from_k_to_T[34]
   - 2598.202362885735 R[34,0] - 2308.422844466397 R[34,1]
   - 2018.904933544298 R[34,2] - 1729.125415118294 R[34,3]
   - 1444.348148751684 R[34,4] - 1154.568630324523 R[34,5]
   - 868.4659045994895 R[34,6] - 579.5590368543213 R[34,7] = 0
 Constr_Left_0_diff_aux_34: Sold_0[34] - Q0_var[34] + Left_0_aux_34 = 0
 Constr_Lost_0_diff_aux_34: - Total_Demand_Up_to_K_minus_1[34] + Q0_var[34]
   + Lost_0_aux_34 = 0
 Constr_Q1_plus_left_34: - Left_0[34] + Q1_plus_left[34] - Q1_var[34] = 0
 Constr_Left_1_diff_aux_34: Sold_1[34] - Q1_plus_left[34] + Left_1_aux_34
   = 0
 Constr_Lost_1_diff_aux_34: - Total_Demand_from_k_to_T[34]
   + Q1_plus_left[34] + Lost_1_aux_34 = 0
 Profit_Constraint_34: - 600 Sold_0[34] + 600 Lost_0[34] - 600 Sold_1[34]
   + 400 Left_1[34] + 600 Lost_1[34] + profits_vars[34] = 0
 R1085: - 17.39692274694594 alphas[0] - 267.2265230377156 alphas[2]
   - alphas[3] + f_var[35] = 0
 Q0_upper_bound_35:
   - 2806.409713277834 Fraction_for_second_order_amount[35] + Q0_var[35]
   = 0
 tau_computation_35_0: - 17.39692274694594 betas[0,0]
   - 267.2265230377156 betas[0,2] - betas[0,3] + tau[35,0] = 0
 tau_computation_35_1: - 17.39692274694594 betas[1,0]
   - 267.2265230377156 betas[1,2] - betas[1,3] + tau[35,1] = 0
 tau_computation_35_2: - 17.39692274694594 betas[2,0]
   - 267.2265230377156 betas[2,2] - betas[2,3] + tau[35,2] = 0
 tau_computation_35_3: - 17.39692274694594 betas[3,0]
   - 267.2265230377156 betas[3,2] - betas[3,3] + tau[35,3] = 0
 tau_computation_35_4: - 17.39692274694594 betas[4,0]
   - 267.2265230377156 betas[4,2] - betas[4,3] + tau[35,4] = 0
 tau_computation_35_5: - 17.39692274694594 betas[5,0]
   - 267.2265230377156 betas[5,2] - betas[5,3] + tau[35,5] = 0
 tau_computation_35_6: - 17.39692274694594 betas[6,0]
   - 267.2265230377156 betas[6,2] - betas[6,3] + tau[35,6] = 0
 tau_computation_35_7: - 17.39692274694594 betas[7,0]
   - 267.2265230377156 betas[7,2] - betas[7,3] + tau[35,7] = 0
 tau_computation_35_8: - 17.39692274694594 betas[8,0]
   - 267.2265230377156 betas[8,2] - betas[8,3] + tau[35,8] = 0
 tau_gap_35_0: tau[35,0] - 5e+06 R[35,0] - tau_star_35 <= -0.001
 tau_gap_35_1: tau[35,1] - 5e+06 R[35,1] - tau_star_35 <= -0.001
 tau_gap_35_2: tau[35,2] - 5e+06 R[35,2] - tau_star_35 <= -0.001
 tau_gap_35_3: tau[35,3] - 5e+06 R[35,3] - tau_star_35 <= -0.001
 tau_gap_35_4: tau[35,4] - 5e+06 R[35,4] - tau_star_35 <= -0.001
 tau_gap_35_5: tau[35,5] - 5e+06 R[35,5] - tau_star_35 <= -0.001
 tau_gap_35_6: tau[35,6] - 5e+06 R[35,6] - tau_star_35 <= -0.001
 tau_gap_35_7: tau[35,7] - 5e+06 R[35,7] - tau_star_35 <= -0.001
 tau_gap_35_8: tau[35,8] - 5e+06 R[35,8] - tau_star_35 <= -0.001
 one_R_35: R[35,0] + R[35,1] + R[35,2] + R[35,3] + R[35,4] + R[35,5]
   + R[35,6] + R[35,7] + R[35,8] = 1
 Define_Q_hat_35: - 2684.184972111627 R[35,0] - 2673.306221809802 R[35,1]
   - 2673.269998878086 R[35,2] - 2673.269999015963 R[35,3]
   - 2673.050679433747 R[35,4] - 2673.050679464269 R[35,5]
   - 2673.840264989188 R[35,6] - 2674.047905794762 R[35,7] + Q_hat[35] = 0
 Adjust_Q_hat_35: Q0_var[35] - Q_hat[35] + Q_hat_adjusted[35] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_35: Total_Demand_Up_to_K_minus_1[35]
   - 267.4532435897494 R[35,0] - 534.6797666147072 R[35,1]
   - 802.3257980033356 R[35,2] - 1069.552321050063 R[35,3]
   - 1336.668371936818 R[35,4] - 1603.894894974534 R[35,5]
   - 1871.756498044617 R[35,6] - 2139.594859135901 R[35,7] = 0
 Constr_Total_Demand_from_K_to_T_35: Total_Demand_from_k_to_T[35]
   - 2406.594661621583 R[35,0] - 2139.368138596626 R[35,1]
   - 1871.722107207997 R[35,2] - 1604.49558416127 R[35,3]
   - 1337.379533274514 R[35,4] - 1070.153010236799 R[35,5]
   - 802.2914071667153 R[35,6] - 534.4530460754312 R[35,7] = 0
 Constr_Left_0_diff_aux_35: Sold_0[35] - Q0_var[35] + Left_0_aux_35 = 0
 Constr_Lost_0_diff_aux_35: - Total_Demand_Up_to_K_minus_1[35] + Q0_var[35]
   + Lost_0_aux_35 = 0
 Constr_Q1_plus_left_35: - Left_0[35] + Q1_plus_left[35] - Q1_var[35] = 0
 Constr_Left_1_diff_aux_35: Sold_1[35] - Q1_plus_left[35] + Left_1_aux_35
   = 0
 Constr_Lost_1_diff_aux_35: - Total_Demand_from_k_to_T[35]
   + Q1_plus_left[35] + Lost_1_aux_35 = 0
 Profit_Constraint_35: - 600 Sold_0[35] + 600 Lost_0[35] - 600 Sold_1[35]
   + 400 Left_1[35] + 600 Lost_1[35] + profits_vars[35] = 0
 R1116: - 19.57431543916691 alphas[0] - 288.5436378284324 alphas[2]
   - alphas[3] + f_var[36] = 0
 Q0_upper_bound_36:
   - 2806.409713277834 Fraction_for_second_order_amount[36] + Q0_var[36]
   = 0
 tau_computation_36_0: - 19.57431543916691 betas[0,0]
   - 288.5436378284324 betas[0,2] - betas[0,3] + tau[36,0] = 0
 tau_computation_36_1: - 19.57431543916691 betas[1,0]
   - 288.5436378284324 betas[1,2] - betas[1,3] + tau[36,1] = 0
 tau_computation_36_2: - 19.57431543916691 betas[2,0]
   - 288.5436378284324 betas[2,2] - betas[2,3] + tau[36,2] = 0
 tau_computation_36_3: - 19.57431543916691 betas[3,0]
   - 288.5436378284324 betas[3,2] - betas[3,3] + tau[36,3] = 0
 tau_computation_36_4: - 19.57431543916691 betas[4,0]
   - 288.5436378284324 betas[4,2] - betas[4,3] + tau[36,4] = 0
 tau_computation_36_5: - 19.57431543916691 betas[5,0]
   - 288.5436378284324 betas[5,2] - betas[5,3] + tau[36,5] = 0
 tau_computation_36_6: - 19.57431543916691 betas[6,0]
   - 288.5436378284324 betas[6,2] - betas[6,3] + tau[36,6] = 0
 tau_computation_36_7: - 19.57431543916691 betas[7,0]
   - 288.5436378284324 betas[7,2] - betas[7,3] + tau[36,7] = 0
 tau_computation_36_8: - 19.57431543916691 betas[8,0]
   - 288.5436378284324 betas[8,2] - betas[8,3] + tau[36,8] = 0
 tau_gap_36_0: tau[36,0] - 5e+06 R[36,0] - tau_star_36 <= -0.001
 tau_gap_36_1: tau[36,1] - 5e+06 R[36,1] - tau_star_36 <= -0.001
 tau_gap_36_2: tau[36,2] - 5e+06 R[36,2] - tau_star_36 <= -0.001
 tau_gap_36_3: tau[36,3] - 5e+06 R[36,3] - tau_star_36 <= -0.001
 tau_gap_36_4: tau[36,4] - 5e+06 R[36,4] - tau_star_36 <= -0.001
 tau_gap_36_5: tau[36,5] - 5e+06 R[36,5] - tau_star_36 <= -0.001
 tau_gap_36_6: tau[36,6] - 5e+06 R[36,6] - tau_star_36 <= -0.001
 tau_gap_36_7: tau[36,7] - 5e+06 R[36,7] - tau_star_36 <= -0.001
 tau_gap_36_8: tau[36,8] - 5e+06 R[36,8] - tau_star_36 <= -0.001
 one_R_36: R[36,0] + R[36,1] + R[36,2] + R[36,3] + R[36,4] + R[36,5]
   + R[36,6] + R[36,7] + R[36,8] = 1
 Define_Q_hat_36: - 2876.171085535668 R[36,0] - 2883.677404887365 R[36,1]
   - 2883.760963073798 R[36,2] - 2883.760963431898 R[36,3]
   - 2877.966990355983 R[36,4] - 2877.966990479671 R[36,5]
   - 2873.960285532614 R[36,6] - 2869.945264297037 R[36,7] + Q_hat[36] = 0
 Adjust_Q_hat_36: Q0_var[36] - Q_hat[36] + Q_hat_adjusted[36] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_36: Total_Demand_Up_to_K_minus_1[36]
   - 288.0617617419431 R[36,0] - 576.6053994597639 R[36,1]
   - 864.1522224235844 R[36,2] - 1152.695860258711 R[36,3]
   - 1436.816206214564 R[36,4] - 1725.359844042996 R[36,5]
   - 2010.045334274674 R[36,6] - 2292.857988035094 R[36,7] = 0
 Constr_Total_Demand_from_K_to_T_36: Total_Demand_from_k_to_T[36]
   - 2581.883501950016 R[36,0] - 2293.339864232194 R[36,1]
   - 2005.793041268374 R[36,2] - 1717.249403433248 R[36,3]
   - 1433.129057477395 R[36,4] - 1144.585419648962 R[36,5]
   - 859.8999294172843 R[36,6] - 577.0872756568648 R[36,7] = 0
 Constr_Left_0_diff_aux_36: Sold_0[36] - Q0_var[36] + Left_0_aux_36 = 0
 Constr_Lost_0_diff_aux_36: - Total_Demand_Up_to_K_minus_1[36] + Q0_var[36]
   + Lost_0_aux_36 = 0
 Constr_Q1_plus_left_36: - Left_0[36] + Q1_plus_left[36] - Q1_var[36] = 0
 Constr_Left_1_diff_aux_36: Sold_1[36] - Q1_plus_left[36] + Left_1_aux_36
   = 0
 Constr_Lost_1_diff_aux_36: - Total_Demand_from_k_to_T[36]
   + Q1_plus_left[36] + Lost_1_aux_36 = 0
 Profit_Constraint_36: - 600 Sold_0[36] + 600 Lost_0[36] - 600 Sold_1[36]
   + 400 Left_1[36] + 600 Lost_1[36] + profits_vars[36] = 0
 R1147: - 15.2467447339422 alphas[0] - 286.7946948403866 alphas[2]
   - alphas[3] + f_var[37] = 0
 Q0_upper_bound_37:
   - 2806.409713277834 Fraction_for_second_order_amount[37] + Q0_var[37]
   = 0
 tau_computation_37_0: - 15.2467447339422 betas[0,0]
   - 286.7946948403866 betas[0,2] - betas[0,3] + tau[37,0] = 0
 tau_computation_37_1: - 15.2467447339422 betas[1,0]
   - 286.7946948403866 betas[1,2] - betas[1,3] + tau[37,1] = 0
 tau_computation_37_2: - 15.2467447339422 betas[2,0]
   - 286.7946948403866 betas[2,2] - betas[2,3] + tau[37,2] = 0
 tau_computation_37_3: - 15.2467447339422 betas[3,0]
   - 286.7946948403866 betas[3,2] - betas[3,3] + tau[37,3] = 0
 tau_computation_37_4: - 15.2467447339422 betas[4,0]
   - 286.7946948403866 betas[4,2] - betas[4,3] + tau[37,4] = 0
 tau_computation_37_5: - 15.2467447339422 betas[5,0]
   - 286.7946948403866 betas[5,2] - betas[5,3] + tau[37,5] = 0
 tau_computation_37_6: - 15.2467447339422 betas[6,0]
   - 286.7946948403866 betas[6,2] - betas[6,3] + tau[37,6] = 0
 tau_computation_37_7: - 15.2467447339422 betas[7,0]
   - 286.7946948403866 betas[7,2] - betas[7,3] + tau[37,7] = 0
 tau_computation_37_8: - 15.2467447339422 betas[8,0]
   - 286.7946948403866 betas[8,2] - betas[8,3] + tau[37,8] = 0
 tau_gap_37_0: tau[37,0] - 5e+06 R[37,0] - tau_star_37 <= -0.001
 tau_gap_37_1: tau[37,1] - 5e+06 R[37,1] - tau_star_37 <= -0.001
 tau_gap_37_2: tau[37,2] - 5e+06 R[37,2] - tau_star_37 <= -0.001
 tau_gap_37_3: tau[37,3] - 5e+06 R[37,3] - tau_star_37 <= -0.001
 tau_gap_37_4: tau[37,4] - 5e+06 R[37,4] - tau_star_37 <= -0.001
 tau_gap_37_5: tau[37,5] - 5e+06 R[37,5] - tau_star_37 <= -0.001
 tau_gap_37_6: tau[37,6] - 5e+06 R[37,6] - tau_star_37 <= -0.001
 tau_gap_37_7: tau[37,7] - 5e+06 R[37,7] - tau_star_37 <= -0.001
 tau_gap_37_8: tau[37,8] - 5e+06 R[37,8] - tau_star_37 <= -0.001
 one_R_37: R[37,0] + R[37,1] + R[37,2] + R[37,3] + R[37,4] + R[37,5]
   + R[37,6] + R[37,7] + R[37,8] = 1
 Define_Q_hat_37: - 2840.589593672123 R[37,0] - 2863.472698981262 R[37,1]
   - 2863.407794294122 R[37,2] - 2863.407794337635 R[37,3]
   - 2865.07977781642 R[37,4] - 2865.079777820807 R[37,5]
   - 2865.527157729642 R[37,6] - 2863.853516860442 R[37,7] + Q_hat[37] = 0
 Adjust_Q_hat_37: Q0_var[37] - Q_hat[37] + Q_hat_adjusted[37] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_37: Total_Demand_Up_to_K_minus_1[37]
   - 284.2423093457566 R[37,0] - 571.0370041492147 R[37,1]
   - 857.1229325070123 R[37,2] - 1143.917627348108 R[37,3]
   - 1431.233932347807 R[37,4] - 1718.028627188194 R[37,5]
   - 2005.92700236001 R[37,6] - 2290.264126569021 R[37,7] = 0
 Constr_Total_Demand_from_K_to_T_37: Total_Demand_from_k_to_T[37]
   - 2579.611206904037 R[37,0] - 2292.816512100579 R[37,1]
   - 2006.730583742782 R[37,2] - 1719.935888901686 R[37,3]
   - 1432.619583901987 R[37,4] - 1145.8248890616 R[37,5]
   - 857.9265138897845 R[37,6] - 573.5893896807733 R[37,7] = 0
 Constr_Left_0_diff_aux_37: Sold_0[37] - Q0_var[37] + Left_0_aux_37 = 0
 Constr_Lost_0_diff_aux_37: - Total_Demand_Up_to_K_minus_1[37] + Q0_var[37]
   + Lost_0_aux_37 = 0
 Constr_Q1_plus_left_37: - Left_0[37] + Q1_plus_left[37] - Q1_var[37] = 0
 Constr_Left_1_diff_aux_37: Sold_1[37] - Q1_plus_left[37] + Left_1_aux_37
   = 0
 Constr_Lost_1_diff_aux_37: - Total_Demand_from_k_to_T[37]
   + Q1_plus_left[37] + Lost_1_aux_37 = 0
 Profit_Constraint_37: - 600 Sold_0[37] + 600 Lost_0[37] - 600 Sold_1[37]
   + 400 Left_1[37] + 600 Lost_1[37] + profits_vars[37] = 0
 R1178: - 16.46444282513507 alphas[0] - 257.075324281095 alphas[2]
   - alphas[3] + f_var[38] = 0
 Q0_upper_bound_38:
   - 2806.409713277834 Fraction_for_second_order_amount[38] + Q0_var[38]
   = 0
 tau_computation_38_0: - 16.46444282513507 betas[0,0]
   - 257.075324281095 betas[0,2] - betas[0,3] + tau[38,0] = 0
 tau_computation_38_1: - 16.46444282513507 betas[1,0]
   - 257.075324281095 betas[1,2] - betas[1,3] + tau[38,1] = 0
 tau_computation_38_2: - 16.46444282513507 betas[2,0]
   - 257.075324281095 betas[2,2] - betas[2,3] + tau[38,2] = 0
 tau_computation_38_3: - 16.46444282513507 betas[3,0]
   - 257.075324281095 betas[3,2] - betas[3,3] + tau[38,3] = 0
 tau_computation_38_4: - 16.46444282513507 betas[4,0]
   - 257.075324281095 betas[4,2] - betas[4,3] + tau[38,4] = 0
 tau_computation_38_5: - 16.46444282513507 betas[5,0]
   - 257.075324281095 betas[5,2] - betas[5,3] + tau[38,5] = 0
 tau_computation_38_6: - 16.46444282513507 betas[6,0]
   - 257.075324281095 betas[6,2] - betas[6,3] + tau[38,6] = 0
 tau_computation_38_7: - 16.46444282513507 betas[7,0]
   - 257.075324281095 betas[7,2] - betas[7,3] + tau[38,7] = 0
 tau_computation_38_8: - 16.46444282513507 betas[8,0]
   - 257.075324281095 betas[8,2] - betas[8,3] + tau[38,8] = 0
 tau_gap_38_0: tau[38,0] - 5e+06 R[38,0] - tau_star_38 <= -0.001
 tau_gap_38_1: tau[38,1] - 5e+06 R[38,1] - tau_star_38 <= -0.001
 tau_gap_38_2: tau[38,2] - 5e+06 R[38,2] - tau_star_38 <= -0.001
 tau_gap_38_3: tau[38,3] - 5e+06 R[38,3] - tau_star_38 <= -0.001
 tau_gap_38_4: tau[38,4] - 5e+06 R[38,4] - tau_star_38 <= -0.001
 tau_gap_38_5: tau[38,5] - 5e+06 R[38,5] - tau_star_38 <= -0.001
 tau_gap_38_6: tau[38,6] - 5e+06 R[38,6] - tau_star_38 <= -0.001
 tau_gap_38_7: tau[38,7] - 5e+06 R[38,7] - tau_star_38 <= -0.001
 tau_gap_38_8: tau[38,8] - 5e+06 R[38,8] - tau_star_38 <= -0.001
 one_R_38: R[38,0] + R[38,1] + R[38,2] + R[38,3] + R[38,4] + R[38,5]
   + R[38,6] + R[38,7] + R[38,8] = 1
 Define_Q_hat_38: - 2546.523809681801 R[38,0] - 2566.260806213196 R[38,1]
   - 2566.182074913926 R[38,2] - 2566.182073796439 R[38,3]
   - 2567.823176238689 R[38,4] - 2567.823175927464 R[38,5]
   - 2565.621429042018 R[38,6] - 2563.58021465701 R[38,7] + Q_hat[38] = 0
 Adjust_Q_hat_38: Q0_var[38] - Q_hat[38] + Q_hat_adjusted[38] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_38: Total_Demand_Up_to_K_minus_1[38]
   - 252.6761718828959 R[38,0] - 509.751496397995 R[38,1]
   - 765.7561401822818 R[38,2] - 1022.831464469822 R[38,3]
   - 1281.547401098531 R[38,4] - 1538.622725379626 R[38,5]
   - 1793.840247075973 R[38,6] - 2049.42956525237 R[38,7] = 0
 Constr_Total_Demand_from_K_to_T_38: Total_Demand_from_k_to_T[38]
   - 2310.904041931664 R[38,0] - 2053.828717416565 R[38,1]
   - 1797.824073632278 R[38,2] - 1540.748749344738 R[38,3]
   - 1282.032812716029 R[38,4] - 1024.957488434934 R[38,5]
   - 769.7399667385866 R[38,6] - 514.15064856219 R[38,7] = 0
 Constr_Left_0_diff_aux_38: Sold_0[38] - Q0_var[38] + Left_0_aux_38 = 0
 Constr_Lost_0_diff_aux_38: - Total_Demand_Up_to_K_minus_1[38] + Q0_var[38]
   + Lost_0_aux_38 = 0
 Constr_Q1_plus_left_38: - Left_0[38] + Q1_plus_left[38] - Q1_var[38] = 0
 Constr_Left_1_diff_aux_38: Sold_1[38] - Q1_plus_left[38] + Left_1_aux_38
   = 0
 Constr_Lost_1_diff_aux_38: - Total_Demand_from_k_to_T[38]
   + Q1_plus_left[38] + Lost_1_aux_38 = 0
 Profit_Constraint_38: - 600 Sold_0[38] + 600 Lost_0[38] - 600 Sold_1[38]
   + 400 Left_1[38] + 600 Lost_1[38] + profits_vars[38] = 0
 R1209: - 18.57526298732584 alphas[0] - 293.2972734283239 alphas[2]
   - alphas[3] + f_var[39] = 0
 Q0_upper_bound_39:
   - 2806.409713277834 Fraction_for_second_order_amount[39] + Q0_var[39]
   = 0
 tau_computation_39_0: - 18.57526298732584 betas[0,0]
   - 293.2972734283239 betas[0,2] - betas[0,3] + tau[39,0] = 0
 tau_computation_39_1: - 18.57526298732584 betas[1,0]
   - 293.2972734283239 betas[1,2] - betas[1,3] + tau[39,1] = 0
 tau_computation_39_2: - 18.57526298732584 betas[2,0]
   - 293.2972734283239 betas[2,2] - betas[2,3] + tau[39,2] = 0
 tau_computation_39_3: - 18.57526298732584 betas[3,0]
   - 293.2972734283239 betas[3,2] - betas[3,3] + tau[39,3] = 0
 tau_computation_39_4: - 18.57526298732584 betas[4,0]
   - 293.2972734283239 betas[4,2] - betas[4,3] + tau[39,4] = 0
 tau_computation_39_5: - 18.57526298732584 betas[5,0]
   - 293.2972734283239 betas[5,2] - betas[5,3] + tau[39,5] = 0
 tau_computation_39_6: - 18.57526298732584 betas[6,0]
   - 293.2972734283239 betas[6,2] - betas[6,3] + tau[39,6] = 0
 tau_computation_39_7: - 18.57526298732584 betas[7,0]
   - 293.2972734283239 betas[7,2] - betas[7,3] + tau[39,7] = 0
 tau_computation_39_8: - 18.57526298732584 betas[8,0]
   - 293.2972734283239 betas[8,2] - betas[8,3] + tau[39,8] = 0
 tau_gap_39_0: tau[39,0] - 5e+06 R[39,0] - tau_star_39 <= -0.001
 tau_gap_39_1: tau[39,1] - 5e+06 R[39,1] - tau_star_39 <= -0.001
 tau_gap_39_2: tau[39,2] - 5e+06 R[39,2] - tau_star_39 <= -0.001
 tau_gap_39_3: tau[39,3] - 5e+06 R[39,3] - tau_star_39 <= -0.001
 tau_gap_39_4: tau[39,4] - 5e+06 R[39,4] - tau_star_39 <= -0.001
 tau_gap_39_5: tau[39,5] - 5e+06 R[39,5] - tau_star_39 <= -0.001
 tau_gap_39_6: tau[39,6] - 5e+06 R[39,6] - tau_star_39 <= -0.001
 tau_gap_39_7: tau[39,7] - 5e+06 R[39,7] - tau_star_39 <= -0.001
 tau_gap_39_8: tau[39,8] - 5e+06 R[39,8] - tau_star_39 <= -0.001
 one_R_39: R[39,0] + R[39,1] + R[39,2] + R[39,3] + R[39,4] + R[39,5]
   + R[39,6] + R[39,7] + R[39,8] = 1
 Define_Q_hat_39: - 2931.783280778551 R[39,0] - 2932.490323956704 R[39,1]
   - 2932.353330595464 R[39,2] - 2932.353330636576 R[39,3]
   - 2929.922531689855 R[39,4] - 2929.922531719056 R[39,5]
   - 2922.989224690283 R[39,6] - 2923.922238012305 R[39,7] + Q_hat[39] = 0
 Adjust_Q_hat_39: Q0_var[39] - Q_hat[39] + Q_hat_adjusted[39] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_39: Total_Demand_Up_to_K_minus_1[39]
   - 294.0313858710328 R[39,0] - 587.3286593172866 R[39,1]
   - 881.7598311365537 R[39,2] - 1175.057104568122 R[39,3]
   - 1465.272018178231 R[39,4] - 1758.569291606555 R[39,5]
   - 2045.411218646631 R[39,6] - 2337.327690462656 R[39,7] = 0
 Constr_Total_Demand_from_K_to_T_39: Total_Demand_from_k_to_T[39]
   - 2629.890851448271 R[39,0] - 2336.593578002018 R[39,1]
   - 2042.162406182751 R[39,2] - 1748.865132751182 R[39,3]
   - 1458.650219141073 R[39,4] - 1165.352945712749 R[39,5]
   - 878.5110186726736 R[39,6] - 586.5945468566478 R[39,7] = 0
 Constr_Left_0_diff_aux_39: Sold_0[39] - Q0_var[39] + Left_0_aux_39 = 0
 Constr_Lost_0_diff_aux_39: - Total_Demand_Up_to_K_minus_1[39] + Q0_var[39]
   + Lost_0_aux_39 = 0
 Constr_Q1_plus_left_39: - Left_0[39] + Q1_plus_left[39] - Q1_var[39] = 0
 Constr_Left_1_diff_aux_39: Sold_1[39] - Q1_plus_left[39] + Left_1_aux_39
   = 0
 Constr_Lost_1_diff_aux_39: - Total_Demand_from_k_to_T[39]
   + Q1_plus_left[39] + Lost_1_aux_39 = 0
 Profit_Constraint_39: - 600 Sold_0[39] + 600 Lost_0[39] - 600 Sold_1[39]
   + 400 Left_1[39] + 600 Lost_1[39] + profits_vars[39] = 0
 R1240: - 17.09054605874 alphas[0] - 272.0660735090206 alphas[2]
   - alphas[3] + f_var[40] = 0
 Q0_upper_bound_40:
   - 2806.409713277834 Fraction_for_second_order_amount[40] + Q0_var[40]
   = 0
 tau_computation_40_0: - 17.09054605874 betas[0,0]
   - 272.0660735090206 betas[0,2] - betas[0,3] + tau[40,0] = 0
 tau_computation_40_1: - 17.09054605874 betas[1,0]
   - 272.0660735090206 betas[1,2] - betas[1,3] + tau[40,1] = 0
 tau_computation_40_2: - 17.09054605874 betas[2,0]
   - 272.0660735090206 betas[2,2] - betas[2,3] + tau[40,2] = 0
 tau_computation_40_3: - 17.09054605874 betas[3,0]
   - 272.0660735090206 betas[3,2] - betas[3,3] + tau[40,3] = 0
 tau_computation_40_4: - 17.09054605874 betas[4,0]
   - 272.0660735090206 betas[4,2] - betas[4,3] + tau[40,4] = 0
 tau_computation_40_5: - 17.09054605874 betas[5,0]
   - 272.0660735090206 betas[5,2] - betas[5,3] + tau[40,5] = 0
 tau_computation_40_6: - 17.09054605874 betas[6,0]
   - 272.0660735090206 betas[6,2] - betas[6,3] + tau[40,6] = 0
 tau_computation_40_7: - 17.09054605874 betas[7,0]
   - 272.0660735090206 betas[7,2] - betas[7,3] + tau[40,7] = 0
 tau_computation_40_8: - 17.09054605874 betas[8,0]
   - 272.0660735090206 betas[8,2] - betas[8,3] + tau[40,8] = 0
 tau_gap_40_0: tau[40,0] - 5e+06 R[40,0] - tau_star_40 <= -0.001
 tau_gap_40_1: tau[40,1] - 5e+06 R[40,1] - tau_star_40 <= -0.001
 tau_gap_40_2: tau[40,2] - 5e+06 R[40,2] - tau_star_40 <= -0.001
 tau_gap_40_3: tau[40,3] - 5e+06 R[40,3] - tau_star_40 <= -0.001
 tau_gap_40_4: tau[40,4] - 5e+06 R[40,4] - tau_star_40 <= -0.001
 tau_gap_40_5: tau[40,5] - 5e+06 R[40,5] - tau_star_40 <= -0.001
 tau_gap_40_6: tau[40,6] - 5e+06 R[40,6] - tau_star_40 <= -0.001
 tau_gap_40_7: tau[40,7] - 5e+06 R[40,7] - tau_star_40 <= -0.001
 tau_gap_40_8: tau[40,8] - 5e+06 R[40,8] - tau_star_40 <= -0.001
 one_R_40: R[40,0] + R[40,1] + R[40,2] + R[40,3] + R[40,4] + R[40,5]
   + R[40,6] + R[40,7] + R[40,8] = 1
 Define_Q_hat_40: - 2710.047291890935 R[40,0] - 2718.433909251411 R[40,1]
   - 2718.171845146542 R[40,2] - 2718.171845094992 R[40,3]
   - 2716.231972537885 R[40,4] - 2716.2319725543 R[40,5]
   - 2716.677340958836 R[40,6] - 2716.165684349067 R[40,7] + Q_hat[40] = 0
 Adjust_Q_hat_40: Q0_var[40] - Q_hat[40] + Q_hat_adjusted[40] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_40: Total_Demand_Up_to_K_minus_1[40]
   - 270.2294031625645 R[40,0] - 542.2954767105828 R[40,1]
   - 815.5937696405474 R[40,2] - 1087.659843146888 R[40,3]
   - 1357.424527283035 R[40,4] - 1629.490600792055 R[40,5]
   - 1900.171817323263 R[40,6] - 2172.033536739095 R[40,7] = 0
 Constr_Total_Demand_from_K_to_T_40: Total_Demand_from_k_to_T[40]
   - 2445.936280594572 R[40,0] - 2173.870207046553 R[40,1]
   - 1900.571914116589 R[40,2] - 1628.505840610248 R[40,3]
   - 1358.741156474101 R[40,4] - 1086.67508296508 R[40,5]
   - 815.9938664338736 R[40,6] - 544.1321470180411 R[40,7] = 0
 Constr_Left_0_diff_aux_40: Sold_0[40] - Q0_var[40] + Left_0_aux_40 = 0
 Constr_Lost_0_diff_aux_40: - Total_Demand_Up_to_K_minus_1[40] + Q0_var[40]
   + Lost_0_aux_40 = 0
 Constr_Q1_plus_left_40: - Left_0[40] + Q1_plus_left[40] - Q1_var[40] = 0
 Constr_Left_1_diff_aux_40: Sold_1[40] - Q1_plus_left[40] + Left_1_aux_40
   = 0
 Constr_Lost_1_diff_aux_40: - Total_Demand_from_k_to_T[40]
   + Q1_plus_left[40] + Lost_1_aux_40 = 0
 Profit_Constraint_40: - 600 Sold_0[40] + 600 Lost_0[40] - 600 Sold_1[40]
   + 400 Left_1[40] + 600 Lost_1[40] + profits_vars[40] = 0
 R1271: - 15.86475677135578 alphas[0] - 274.3205224444333 alphas[2]
   - alphas[3] + f_var[41] = 0
 Q0_upper_bound_41:
   - 2806.409713277834 Fraction_for_second_order_amount[41] + Q0_var[41]
   = 0
 tau_computation_41_0: - 15.86475677135578 betas[0,0]
   - 274.3205224444333 betas[0,2] - betas[0,3] + tau[41,0] = 0
 tau_computation_41_1: - 15.86475677135578 betas[1,0]
   - 274.3205224444333 betas[1,2] - betas[1,3] + tau[41,1] = 0
 tau_computation_41_2: - 15.86475677135578 betas[2,0]
   - 274.3205224444333 betas[2,2] - betas[2,3] + tau[41,2] = 0
 tau_computation_41_3: - 15.86475677135578 betas[3,0]
   - 274.3205224444333 betas[3,2] - betas[3,3] + tau[41,3] = 0
 tau_computation_41_4: - 15.86475677135578 betas[4,0]
   - 274.3205224444333 betas[4,2] - betas[4,3] + tau[41,4] = 0
 tau_computation_41_5: - 15.86475677135578 betas[5,0]
   - 274.3205224444333 betas[5,2] - betas[5,3] + tau[41,5] = 0
 tau_computation_41_6: - 15.86475677135578 betas[6,0]
   - 274.3205224444333 betas[6,2] - betas[6,3] + tau[41,6] = 0
 tau_computation_41_7: - 15.86475677135578 betas[7,0]
   - 274.3205224444333 betas[7,2] - betas[7,3] + tau[41,7] = 0
 tau_computation_41_8: - 15.86475677135578 betas[8,0]
   - 274.3205224444333 betas[8,2] - betas[8,3] + tau[41,8] = 0
 tau_gap_41_0: tau[41,0] - 5e+06 R[41,0] - tau_star_41 <= -0.001
 tau_gap_41_1: tau[41,1] - 5e+06 R[41,1] - tau_star_41 <= -0.001
 tau_gap_41_2: tau[41,2] - 5e+06 R[41,2] - tau_star_41 <= -0.001
 tau_gap_41_3: tau[41,3] - 5e+06 R[41,3] - tau_star_41 <= -0.001
 tau_gap_41_4: tau[41,4] - 5e+06 R[41,4] - tau_star_41 <= -0.001
 tau_gap_41_5: tau[41,5] - 5e+06 R[41,5] - tau_star_41 <= -0.001
 tau_gap_41_6: tau[41,6] - 5e+06 R[41,6] - tau_star_41 <= -0.001
 tau_gap_41_7: tau[41,7] - 5e+06 R[41,7] - tau_star_41 <= -0.001
 tau_gap_41_8: tau[41,8] - 5e+06 R[41,8] - tau_star_41 <= -0.001
 one_R_41: R[41,0] + R[41,1] + R[41,2] + R[41,3] + R[41,4] + R[41,5]
   + R[41,6] + R[41,7] + R[41,8] = 1
 Define_Q_hat_41: - 2737.688590528448 R[41,0] - 2741.771949152561 R[41,1]
   - 2741.789497095336 R[41,2] - 2741.789496703212 R[41,3]
   - 2741.682709770283 R[41,4] - 2741.682709629719 R[41,5]
   - 2745.828884356454 R[41,6] - 2747.516399132017 R[41,7] + Q_hat[41] = 0
 Adjust_Q_hat_41: Q0_var[41] - Q_hat[41] + Q_hat_adjusted[41] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_41: Total_Demand_Up_to_K_minus_1[41]
   - 273.1965250786336 R[41,0] - 547.5170476140147 R[41,1]
   - 821.2186570748579 R[41,2] - 1095.539179540097 R[41,3]
   - 1369.826818196203 R[41,4] - 1644.147340640637 R[41,5]
   - 1922.365897913682 R[41,6] - 2198.875353628673 R[41,7] = 0
 Constr_Total_Demand_from_K_to_T_41: Total_Demand_from_k_to_T[41]
   - 2474.319873438906 R[41,0] - 2199.999350903524 R[41,1]
   - 1926.297741442681 R[41,2] - 1651.977218977442 R[41,3]
   - 1377.689580321336 R[41,4] - 1103.369057876902 R[41,5]
   - 825.1505006038572 R[41,6] - 548.6410448888665 R[41,7] = 0
 Constr_Left_0_diff_aux_41: Sold_0[41] - Q0_var[41] + Left_0_aux_41 = 0
 Constr_Lost_0_diff_aux_41: - Total_Demand_Up_to_K_minus_1[41] + Q0_var[41]
   + Lost_0_aux_41 = 0
 Constr_Q1_plus_left_41: - Left_0[41] + Q1_plus_left[41] - Q1_var[41] = 0
 Constr_Left_1_diff_aux_41: Sold_1[41] - Q1_plus_left[41] + Left_1_aux_41
   = 0
 Constr_Lost_1_diff_aux_41: - Total_Demand_from_k_to_T[41]
   + Q1_plus_left[41] + Lost_1_aux_41 = 0
 Profit_Constraint_41: - 600 Sold_0[41] + 600 Lost_0[41] - 600 Sold_1[41]
   + 400 Left_1[41] + 600 Lost_1[41] + profits_vars[41] = 0
 R1302: - 15.53605372714273 alphas[0] - 272.4184589448998 alphas[2]
   - alphas[3] + f_var[42] = 0
 Q0_upper_bound_42:
   - 2806.409713277834 Fraction_for_second_order_amount[42] + Q0_var[42]
   = 0
 tau_computation_42_0: - 15.53605372714273 betas[0,0]
   - 272.4184589448998 betas[0,2] - betas[0,3] + tau[42,0] = 0
 tau_computation_42_1: - 15.53605372714273 betas[1,0]
   - 272.4184589448998 betas[1,2] - betas[1,3] + tau[42,1] = 0
 tau_computation_42_2: - 15.53605372714273 betas[2,0]
   - 272.4184589448998 betas[2,2] - betas[2,3] + tau[42,2] = 0
 tau_computation_42_3: - 15.53605372714273 betas[3,0]
   - 272.4184589448998 betas[3,2] - betas[3,3] + tau[42,3] = 0
 tau_computation_42_4: - 15.53605372714273 betas[4,0]
   - 272.4184589448998 betas[4,2] - betas[4,3] + tau[42,4] = 0
 tau_computation_42_5: - 15.53605372714273 betas[5,0]
   - 272.4184589448998 betas[5,2] - betas[5,3] + tau[42,5] = 0
 tau_computation_42_6: - 15.53605372714273 betas[6,0]
   - 272.4184589448998 betas[6,2] - betas[6,3] + tau[42,6] = 0
 tau_computation_42_7: - 15.53605372714273 betas[7,0]
   - 272.4184589448998 betas[7,2] - betas[7,3] + tau[42,7] = 0
 tau_computation_42_8: - 15.53605372714273 betas[8,0]
   - 272.4184589448998 betas[8,2] - betas[8,3] + tau[42,8] = 0
 tau_gap_42_0: tau[42,0] - 5e+06 R[42,0] - tau_star_42 <= -0.001
 tau_gap_42_1: tau[42,1] - 5e+06 R[42,1] - tau_star_42 <= -0.001
 tau_gap_42_2: tau[42,2] - 5e+06 R[42,2] - tau_star_42 <= -0.001
 tau_gap_42_3: tau[42,3] - 5e+06 R[42,3] - tau_star_42 <= -0.001
 tau_gap_42_4: tau[42,4] - 5e+06 R[42,4] - tau_star_42 <= -0.001
 tau_gap_42_5: tau[42,5] - 5e+06 R[42,5] - tau_star_42 <= -0.001
 tau_gap_42_6: tau[42,6] - 5e+06 R[42,6] - tau_star_42 <= -0.001
 tau_gap_42_7: tau[42,7] - 5e+06 R[42,7] - tau_star_42 <= -0.001
 tau_gap_42_8: tau[42,8] - 5e+06 R[42,8] - tau_star_42 <= -0.001
 one_R_42: R[42,0] + R[42,1] + R[42,2] + R[42,3] + R[42,4] + R[42,5]
   + R[42,6] + R[42,7] + R[42,8] = 1
 Define_Q_hat_42: - 2719.168732667406 R[42,0] - 2722.794789043072 R[42,1]
   - 2723.186016846942 R[42,2] - 2723.18601682419 R[42,3]
   - 2726.898810333365 R[42,4] - 2726.898810296962 R[42,5]
   - 2726.224638049568 R[42,6] - 2726.995999764103 R[42,7] + Q_hat[42] = 0
 Adjust_Q_hat_42: Q0_var[42] - Q_hat[42] + Q_hat_adjusted[42] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_42: Total_Demand_Up_to_K_minus_1[42]
   - 271.2085332385032 R[42,0] - 543.6269920951473 R[42,1]
   - 812.5720626402402 R[42,2] - 1084.990521590462 R[42,3]
   - 1361.619560250146 R[42,4] - 1634.038019195046 R[42,5]
   - 1909.558265315998 R[42,6] - 2182.15908122972 R[42,7] = 0
 Constr_Total_Demand_from_K_to_T_42: Total_Demand_from_k_to_T[42]
   - 2455.787465881016 R[42,0] - 2183.369007024372 R[42,1]
   - 1914.423936479279 R[42,2] - 1642.005477529057 R[42,3]
   - 1365.376438869374 R[42,4] - 1092.957979924474 R[42,5]
   - 817.4377338035214 R[42,6] - 544.8369178897997 R[42,7] = 0
 Constr_Left_0_diff_aux_42: Sold_0[42] - Q0_var[42] + Left_0_aux_42 = 0
 Constr_Lost_0_diff_aux_42: - Total_Demand_Up_to_K_minus_1[42] + Q0_var[42]
   + Lost_0_aux_42 = 0
 Constr_Q1_plus_left_42: - Left_0[42] + Q1_plus_left[42] - Q1_var[42] = 0
 Constr_Left_1_diff_aux_42: Sold_1[42] - Q1_plus_left[42] + Left_1_aux_42
   = 0
 Constr_Lost_1_diff_aux_42: - Total_Demand_from_k_to_T[42]
   + Q1_plus_left[42] + Lost_1_aux_42 = 0
 Profit_Constraint_42: - 600 Sold_0[42] + 600 Lost_0[42] - 600 Sold_1[42]
   + 400 Left_1[42] + 600 Lost_1[42] + profits_vars[42] = 0
 R1333: - 19.08669555730811 alphas[0] - 278.3923000738754 alphas[2]
   - alphas[3] + f_var[43] = 0
 Q0_upper_bound_43:
   - 2806.409713277834 Fraction_for_second_order_amount[43] + Q0_var[43]
   = 0
 tau_computation_43_0: - 19.08669555730811 betas[0,0]
   - 278.3923000738754 betas[0,2] - betas[0,3] + tau[43,0] = 0
 tau_computation_43_1: - 19.08669555730811 betas[1,0]
   - 278.3923000738754 betas[1,2] - betas[1,3] + tau[43,1] = 0
 tau_computation_43_2: - 19.08669555730811 betas[2,0]
   - 278.3923000738754 betas[2,2] - betas[2,3] + tau[43,2] = 0
 tau_computation_43_3: - 19.08669555730811 betas[3,0]
   - 278.3923000738754 betas[3,2] - betas[3,3] + tau[43,3] = 0
 tau_computation_43_4: - 19.08669555730811 betas[4,0]
   - 278.3923000738754 betas[4,2] - betas[4,3] + tau[43,4] = 0
 tau_computation_43_5: - 19.08669555730811 betas[5,0]
   - 278.3923000738754 betas[5,2] - betas[5,3] + tau[43,5] = 0
 tau_computation_43_6: - 19.08669555730811 betas[6,0]
   - 278.3923000738754 betas[6,2] - betas[6,3] + tau[43,6] = 0
 tau_computation_43_7: - 19.08669555730811 betas[7,0]
   - 278.3923000738754 betas[7,2] - betas[7,3] + tau[43,7] = 0
 tau_computation_43_8: - 19.08669555730811 betas[8,0]
   - 278.3923000738754 betas[8,2] - betas[8,3] + tau[43,8] = 0
 tau_gap_43_0: tau[43,0] - 5e+06 R[43,0] - tau_star_43 <= -0.001
 tau_gap_43_1: tau[43,1] - 5e+06 R[43,1] - tau_star_43 <= -0.001
 tau_gap_43_2: tau[43,2] - 5e+06 R[43,2] - tau_star_43 <= -0.001
 tau_gap_43_3: tau[43,3] - 5e+06 R[43,3] - tau_star_43 <= -0.001
 tau_gap_43_4: tau[43,4] - 5e+06 R[43,4] - tau_star_43 <= -0.001
 tau_gap_43_5: tau[43,5] - 5e+06 R[43,5] - tau_star_43 <= -0.001
 tau_gap_43_6: tau[43,6] - 5e+06 R[43,6] - tau_star_43 <= -0.001
 tau_gap_43_7: tau[43,7] - 5e+06 R[43,7] - tau_star_43 <= -0.001
 tau_gap_43_8: tau[43,8] - 5e+06 R[43,8] - tau_star_43 <= -0.001
 one_R_43: R[43,0] + R[43,1] + R[43,2] + R[43,3] + R[43,4] + R[43,5]
   + R[43,6] + R[43,7] + R[43,8] = 1
 Define_Q_hat_43: - 2825.037420808854 R[43,0] - 2789.481083003436 R[43,1]
   - 2789.808273077458 R[43,2] - 2789.80827294661 R[43,3]
   - 2788.523530272039 R[43,4] - 2788.523530261738 R[43,5]
   - 2785.806175589298 R[43,6] - 2785.720250662843 R[43,7] + Q_hat[43] = 0
 Adjust_Q_hat_43: Q0_var[43] - Q_hat[43] + Q_hat_adjusted[43] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_43: Total_Demand_Up_to_K_minus_1[43]
   - 282.5728800779697 R[43,0] - 560.9651801583938 R[43,1]
   - 838.5417406815418 R[43,2] - 1116.934040754431 R[43,3]
   - 1395.251632896413 R[43,4] - 1673.643932970288 R[43,5]
   - 1951.836000461449 R[43,6] - 2228.935649849009 R[43,7] = 0
 Constr_Total_Demand_from_K_to_T_43: Total_Demand_from_k_to_T[43]
   - 2503.14736991879 R[43,0] - 2224.755069838367 R[43,1]
   - 1947.178509315218 R[43,2] - 1668.786209242329 R[43,3]
   - 1390.468617100347 R[43,4] - 1112.076317026472 R[43,5]
   - 833.8842495353111 R[43,6] - 556.7846001477508 R[43,7] = 0
 Constr_Left_0_diff_aux_43: Sold_0[43] - Q0_var[43] + Left_0_aux_43 = 0
 Constr_Lost_0_diff_aux_43: - Total_Demand_Up_to_K_minus_1[43] + Q0_var[43]
   + Lost_0_aux_43 = 0
 Constr_Q1_plus_left_43: - Left_0[43] + Q1_plus_left[43] - Q1_var[43] = 0
 Constr_Left_1_diff_aux_43: Sold_1[43] - Q1_plus_left[43] + Left_1_aux_43
   = 0
 Constr_Lost_1_diff_aux_43: - Total_Demand_from_k_to_T[43]
   + Q1_plus_left[43] + Lost_1_aux_43 = 0
 Profit_Constraint_43: - 600 Sold_0[43] + 600 Lost_0[43] - 600 Sold_1[43]
   + 400 Left_1[43] + 600 Lost_1[43] + profits_vars[43] = 0
 R1364: - 17.36571489232822 alphas[0] - 281.0584623683527 alphas[2]
   - alphas[3] + f_var[44] = 0
 Q0_upper_bound_44:
   - 2806.409713277834 Fraction_for_second_order_amount[44] + Q0_var[44]
   = 0
 tau_computation_44_0: - 17.36571489232822 betas[0,0]
   - 281.0584623683527 betas[0,2] - betas[0,3] + tau[44,0] = 0
 tau_computation_44_1: - 17.36571489232822 betas[1,0]
   - 281.0584623683527 betas[1,2] - betas[1,3] + tau[44,1] = 0
 tau_computation_44_2: - 17.36571489232822 betas[2,0]
   - 281.0584623683527 betas[2,2] - betas[2,3] + tau[44,2] = 0
 tau_computation_44_3: - 17.36571489232822 betas[3,0]
   - 281.0584623683527 betas[3,2] - betas[3,3] + tau[44,3] = 0
 tau_computation_44_4: - 17.36571489232822 betas[4,0]
   - 281.0584623683527 betas[4,2] - betas[4,3] + tau[44,4] = 0
 tau_computation_44_5: - 17.36571489232822 betas[5,0]
   - 281.0584623683527 betas[5,2] - betas[5,3] + tau[44,5] = 0
 tau_computation_44_6: - 17.36571489232822 betas[6,0]
   - 281.0584623683527 betas[6,2] - betas[6,3] + tau[44,6] = 0
 tau_computation_44_7: - 17.36571489232822 betas[7,0]
   - 281.0584623683527 betas[7,2] - betas[7,3] + tau[44,7] = 0
 tau_computation_44_8: - 17.36571489232822 betas[8,0]
   - 281.0584623683527 betas[8,2] - betas[8,3] + tau[44,8] = 0
 tau_gap_44_0: tau[44,0] - 5e+06 R[44,0] - tau_star_44 <= -0.001
 tau_gap_44_1: tau[44,1] - 5e+06 R[44,1] - tau_star_44 <= -0.001
 tau_gap_44_2: tau[44,2] - 5e+06 R[44,2] - tau_star_44 <= -0.001
 tau_gap_44_3: tau[44,3] - 5e+06 R[44,3] - tau_star_44 <= -0.001
 tau_gap_44_4: tau[44,4] - 5e+06 R[44,4] - tau_star_44 <= -0.001
 tau_gap_44_5: tau[44,5] - 5e+06 R[44,5] - tau_star_44 <= -0.001
 tau_gap_44_6: tau[44,6] - 5e+06 R[44,6] - tau_star_44 <= -0.001
 tau_gap_44_7: tau[44,7] - 5e+06 R[44,7] - tau_star_44 <= -0.001
 tau_gap_44_8: tau[44,8] - 5e+06 R[44,8] - tau_star_44 <= -0.001
 one_R_44: R[44,0] + R[44,1] + R[44,2] + R[44,3] + R[44,4] + R[44,5]
   + R[44,6] + R[44,7] + R[44,8] = 1
 Define_Q_hat_44: - 2864.11573326237 R[44,0] - 2818.030025723105 R[44,1]
   - 2817.752126432732 R[44,2] - 2817.752126862745 R[44,3]
   - 2826.260017518964 R[44,4] - 2826.260017622323 R[44,5]
   - 2820.762184485272 R[44,6] - 2822.082326485779 R[44,7] + Q_hat[44] = 0
 Adjust_Q_hat_44: Q0_var[44] - Q_hat[44] + Q_hat_adjusted[44] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_44: Total_Demand_Up_to_K_minus_1[44]
   - 286.7676944694492 R[44,0] - 567.826156847625 R[44,1]
   - 853.2703031575981 R[44,2] - 1134.328765515302 R[44,3]
   - 1419.863815363425 R[44,4] - 1700.922277731778 R[44,5]
   - 1978.936759486345 R[44,6] - 2259.965401067696 R[44,7] = 0
 Constr_Total_Demand_from_K_to_T_44: Total_Demand_from_k_to_T[44]
   - 2535.314631334952 R[44,0] - 2254.256168956776 R[44,1]
   - 1968.812022646803 R[44,2] - 1687.753560289099 R[44,3]
   - 1402.218510440976 R[44,4] - 1121.160048072623 R[44,5]
   - 843.1455663180566 R[44,6] - 562.1169247367054 R[44,7] = 0
 Constr_Left_0_diff_aux_44: Sold_0[44] - Q0_var[44] + Left_0_aux_44 = 0
 Constr_Lost_0_diff_aux_44: - Total_Demand_Up_to_K_minus_1[44] + Q0_var[44]
   + Lost_0_aux_44 = 0
 Constr_Q1_plus_left_44: - Left_0[44] + Q1_plus_left[44] - Q1_var[44] = 0
 Constr_Left_1_diff_aux_44: Sold_1[44] - Q1_plus_left[44] + Left_1_aux_44
   = 0
 Constr_Lost_1_diff_aux_44: - Total_Demand_from_k_to_T[44]
   + Q1_plus_left[44] + Lost_1_aux_44 = 0
 Profit_Constraint_44: - 600 Sold_0[44] + 600 Lost_0[44] - 600 Sold_1[44]
   + 400 Left_1[44] + 600 Lost_1[44] + profits_vars[44] = 0
 R1395: - 19.41141835959554 alphas[0] - 274.9089782881472 alphas[2]
   - alphas[3] + f_var[45] = 0
 Q0_upper_bound_45:
   - 2806.409713277834 Fraction_for_second_order_amount[45] + Q0_var[45]
   = 0
 tau_computation_45_0: - 19.41141835959554 betas[0,0]
   - 274.9089782881472 betas[0,2] - betas[0,3] + tau[45,0] = 0
 tau_computation_45_1: - 19.41141835959554 betas[1,0]
   - 274.9089782881472 betas[1,2] - betas[1,3] + tau[45,1] = 0
 tau_computation_45_2: - 19.41141835959554 betas[2,0]
   - 274.9089782881472 betas[2,2] - betas[2,3] + tau[45,2] = 0
 tau_computation_45_3: - 19.41141835959554 betas[3,0]
   - 274.9089782881472 betas[3,2] - betas[3,3] + tau[45,3] = 0
 tau_computation_45_4: - 19.41141835959554 betas[4,0]
   - 274.9089782881472 betas[4,2] - betas[4,3] + tau[45,4] = 0
 tau_computation_45_5: - 19.41141835959554 betas[5,0]
   - 274.9089782881472 betas[5,2] - betas[5,3] + tau[45,5] = 0
 tau_computation_45_6: - 19.41141835959554 betas[6,0]
   - 274.9089782881472 betas[6,2] - betas[6,3] + tau[45,6] = 0
 tau_computation_45_7: - 19.41141835959554 betas[7,0]
   - 274.9089782881472 betas[7,2] - betas[7,3] + tau[45,7] = 0
 tau_computation_45_8: - 19.41141835959554 betas[8,0]
   - 274.9089782881472 betas[8,2] - betas[8,3] + tau[45,8] = 0
 tau_gap_45_0: tau[45,0] - 5e+06 R[45,0] - tau_star_45 <= -0.001
 tau_gap_45_1: tau[45,1] - 5e+06 R[45,1] - tau_star_45 <= -0.001
 tau_gap_45_2: tau[45,2] - 5e+06 R[45,2] - tau_star_45 <= -0.001
 tau_gap_45_3: tau[45,3] - 5e+06 R[45,3] - tau_star_45 <= -0.001
 tau_gap_45_4: tau[45,4] - 5e+06 R[45,4] - tau_star_45 <= -0.001
 tau_gap_45_5: tau[45,5] - 5e+06 R[45,5] - tau_star_45 <= -0.001
 tau_gap_45_6: tau[45,6] - 5e+06 R[45,6] - tau_star_45 <= -0.001
 tau_gap_45_7: tau[45,7] - 5e+06 R[45,7] - tau_star_45 <= -0.001
 tau_gap_45_8: tau[45,8] - 5e+06 R[45,8] - tau_star_45 <= -0.001
 one_R_45: R[45,0] + R[45,1] + R[45,2] + R[45,3] + R[45,4] + R[45,5]
   + R[45,6] + R[45,7] + R[45,8] = 1
 Define_Q_hat_45: - 2737.293734289965 R[45,0] - 2746.733502477629 R[45,1]
   - 2747.067009489368 R[45,2] - 2747.067009414824 R[45,3]
   - 2745.363015571852 R[45,4] - 2745.363015545237 R[45,5]
   - 2743.350011273731 R[45,6] - 2742.110461756261 R[45,7] + Q_hat[45] = 0
 Adjust_Q_hat_45: Q0_var[45] - Q_hat[45] + Q_hat_adjusted[45] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_45: Total_Demand_Up_to_K_minus_1[45]
   - 273.1541397112069 R[45,0] - 548.0631179321308 R[45,1]
   - 819.6866115954746 R[45,2] - 1094.595589893824 R[45,3]
   - 1369.537097331199 R[45,4] - 1644.446075619347 R[45,5]
   - 1919.472032653711 R[45,6] - 2192.292504526704 R[45,7] = 0
 Constr_Total_Demand_from_K_to_T_45: Total_Demand_from_k_to_T[45]
   - 2468.956321391792 R[45,0] - 2194.047343170868 R[45,1]
   - 1922.423849507524 R[45,2] - 1647.514871209175 R[45,3]
   - 1372.573363771799 R[45,4] - 1097.664385483652 R[45,5]
   - 822.6384284492871 R[45,6] - 549.8179565762944 R[45,7] = 0
 Constr_Left_0_diff_aux_45: Sold_0[45] - Q0_var[45] + Left_0_aux_45 = 0
 Constr_Lost_0_diff_aux_45: - Total_Demand_Up_to_K_minus_1[45] + Q0_var[45]
   + Lost_0_aux_45 = 0
 Constr_Q1_plus_left_45: - Left_0[45] + Q1_plus_left[45] - Q1_var[45] = 0
 Constr_Left_1_diff_aux_45: Sold_1[45] - Q1_plus_left[45] + Left_1_aux_45
   = 0
 Constr_Lost_1_diff_aux_45: - Total_Demand_from_k_to_T[45]
   + Q1_plus_left[45] + Lost_1_aux_45 = 0
 Profit_Constraint_45: - 600 Sold_0[45] + 600 Lost_0[45] - 600 Sold_1[45]
   + 400 Left_1[45] + 600 Lost_1[45] + profits_vars[45] = 0
 R1426: - 18.66644567158363 alphas[0] - 293.3394271629548 alphas[2]
   - alphas[3] + f_var[46] = 0
 Q0_upper_bound_46:
   - 2806.409713277834 Fraction_for_second_order_amount[46] + Q0_var[46]
   = 0
 tau_computation_46_0: - 18.66644567158363 betas[0,0]
   - 293.3394271629548 betas[0,2] - betas[0,3] + tau[46,0] = 0
 tau_computation_46_1: - 18.66644567158363 betas[1,0]
   - 293.3394271629548 betas[1,2] - betas[1,3] + tau[46,1] = 0
 tau_computation_46_2: - 18.66644567158363 betas[2,0]
   - 293.3394271629548 betas[2,2] - betas[2,3] + tau[46,2] = 0
 tau_computation_46_3: - 18.66644567158363 betas[3,0]
   - 293.3394271629548 betas[3,2] - betas[3,3] + tau[46,3] = 0
 tau_computation_46_4: - 18.66644567158363 betas[4,0]
   - 293.3394271629548 betas[4,2] - betas[4,3] + tau[46,4] = 0
 tau_computation_46_5: - 18.66644567158363 betas[5,0]
   - 293.3394271629548 betas[5,2] - betas[5,3] + tau[46,5] = 0
 tau_computation_46_6: - 18.66644567158363 betas[6,0]
   - 293.3394271629548 betas[6,2] - betas[6,3] + tau[46,6] = 0
 tau_computation_46_7: - 18.66644567158363 betas[7,0]
   - 293.3394271629548 betas[7,2] - betas[7,3] + tau[46,7] = 0
 tau_computation_46_8: - 18.66644567158363 betas[8,0]
   - 293.3394271629548 betas[8,2] - betas[8,3] + tau[46,8] = 0
 tau_gap_46_0: tau[46,0] - 5e+06 R[46,0] - tau_star_46 <= -0.001
 tau_gap_46_1: tau[46,1] - 5e+06 R[46,1] - tau_star_46 <= -0.001
 tau_gap_46_2: tau[46,2] - 5e+06 R[46,2] - tau_star_46 <= -0.001
 tau_gap_46_3: tau[46,3] - 5e+06 R[46,3] - tau_star_46 <= -0.001
 tau_gap_46_4: tau[46,4] - 5e+06 R[46,4] - tau_star_46 <= -0.001
 tau_gap_46_5: tau[46,5] - 5e+06 R[46,5] - tau_star_46 <= -0.001
 tau_gap_46_6: tau[46,6] - 5e+06 R[46,6] - tau_star_46 <= -0.001
 tau_gap_46_7: tau[46,7] - 5e+06 R[46,7] - tau_star_46 <= -0.001
 tau_gap_46_8: tau[46,8] - 5e+06 R[46,8] - tau_star_46 <= -0.001
 one_R_46: R[46,0] + R[46,1] + R[46,2] + R[46,3] + R[46,4] + R[46,5]
   + R[46,6] + R[46,7] + R[46,8] = 1
 Define_Q_hat_46: - 2885.786144699586 R[46,0] - 2926.018864837438 R[46,1]
   - 2925.845950432327 R[46,2] - 2925.84595020722 R[46,3]
   - 2921.591538191495 R[46,4] - 2921.591538175184 R[46,5]
   - 2923.418110172687 R[46,6] - 2922.648053469411 R[46,7] + Q_hat[46] = 0
 Adjust_Q_hat_46: Q0_var[46] - Q_hat[46] + Q_hat_adjusted[46] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_46: Total_Demand_Up_to_K_minus_1[46]
   - 289.0938786892577 R[46,0] - 582.4333058699333 R[46,1]
   - 875.1179644673707 R[46,2] - 1168.45739162472 R[46,3]
   - 1457.479446867486 R[46,4] - 1750.81887403044 R[46,5]
   - 2044.168445616154 R[46,6] - 2335.969198574181 R[46,7] = 0
 Constr_Total_Demand_from_K_to_T_46: Total_Demand_from_k_to_T[46]
   - 2633.554174210833 R[46,0] - 2340.214747030158 R[46,1]
   - 2047.53008843272 R[46,2] - 1754.190661275371 R[46,3]
   - 1465.168606032606 R[46,4] - 1171.829178869651 R[46,5]
   - 878.4796072839376 R[46,6] - 586.6788543259096 R[46,7] = 0
 Constr_Left_0_diff_aux_46: Sold_0[46] - Q0_var[46] + Left_0_aux_46 = 0
 Constr_Lost_0_diff_aux_46: - Total_Demand_Up_to_K_minus_1[46] + Q0_var[46]
   + Lost_0_aux_46 = 0
 Constr_Q1_plus_left_46: - Left_0[46] + Q1_plus_left[46] - Q1_var[46] = 0
 Constr_Left_1_diff_aux_46: Sold_1[46] - Q1_plus_left[46] + Left_1_aux_46
   = 0
 Constr_Lost_1_diff_aux_46: - Total_Demand_from_k_to_T[46]
   + Q1_plus_left[46] + Lost_1_aux_46 = 0
 Profit_Constraint_46: - 600 Sold_0[46] + 600 Lost_0[46] - 600 Sold_1[46]
   + 400 Left_1[46] + 600 Lost_1[46] + profits_vars[46] = 0
 R1457: - 17.04863102815372 alphas[0] - 281.3867378097642 alphas[2]
   - alphas[3] + f_var[47] = 0
 Q0_upper_bound_47:
   - 2806.409713277834 Fraction_for_second_order_amount[47] + Q0_var[47]
   = 0
 tau_computation_47_0: - 17.04863102815372 betas[0,0]
   - 281.3867378097642 betas[0,2] - betas[0,3] + tau[47,0] = 0
 tau_computation_47_1: - 17.04863102815372 betas[1,0]
   - 281.3867378097642 betas[1,2] - betas[1,3] + tau[47,1] = 0
 tau_computation_47_2: - 17.04863102815372 betas[2,0]
   - 281.3867378097642 betas[2,2] - betas[2,3] + tau[47,2] = 0
 tau_computation_47_3: - 17.04863102815372 betas[3,0]
   - 281.3867378097642 betas[3,2] - betas[3,3] + tau[47,3] = 0
 tau_computation_47_4: - 17.04863102815372 betas[4,0]
   - 281.3867378097642 betas[4,2] - betas[4,3] + tau[47,4] = 0
 tau_computation_47_5: - 17.04863102815372 betas[5,0]
   - 281.3867378097642 betas[5,2] - betas[5,3] + tau[47,5] = 0
 tau_computation_47_6: - 17.04863102815372 betas[6,0]
   - 281.3867378097642 betas[6,2] - betas[6,3] + tau[47,6] = 0
 tau_computation_47_7: - 17.04863102815372 betas[7,0]
   - 281.3867378097642 betas[7,2] - betas[7,3] + tau[47,7] = 0
 tau_computation_47_8: - 17.04863102815372 betas[8,0]
   - 281.3867378097642 betas[8,2] - betas[8,3] + tau[47,8] = 0
 tau_gap_47_0: tau[47,0] - 5e+06 R[47,0] - tau_star_47 <= -0.001
 tau_gap_47_1: tau[47,1] - 5e+06 R[47,1] - tau_star_47 <= -0.001
 tau_gap_47_2: tau[47,2] - 5e+06 R[47,2] - tau_star_47 <= -0.001
 tau_gap_47_3: tau[47,3] - 5e+06 R[47,3] - tau_star_47 <= -0.001
 tau_gap_47_4: tau[47,4] - 5e+06 R[47,4] - tau_star_47 <= -0.001
 tau_gap_47_5: tau[47,5] - 5e+06 R[47,5] - tau_star_47 <= -0.001
 tau_gap_47_6: tau[47,6] - 5e+06 R[47,6] - tau_star_47 <= -0.001
 tau_gap_47_7: tau[47,7] - 5e+06 R[47,7] - tau_star_47 <= -0.001
 tau_gap_47_8: tau[47,8] - 5e+06 R[47,8] - tau_star_47 <= -0.001
 one_R_47: R[47,0] + R[47,1] + R[47,2] + R[47,3] + R[47,4] + R[47,5]
   + R[47,6] + R[47,7] + R[47,8] = 1
 Define_Q_hat_47: - 2799.954301489019 R[47,0] - 2811.301913201547 R[47,1]
   - 2810.277598628802 R[47,2] - 2810.277598921286 R[47,3]
   - 2807.748126697502 R[47,4] - 2807.748126797874 R[47,5]
   - 2805.624665214162 R[47,6] - 2800.567793382609 R[47,7] + Q_hat[47] = 0
 Adjust_Q_hat_47: Q0_var[47] - Q_hat[47] + Q_hat_adjusted[47] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_47: Total_Demand_Up_to_K_minus_1[47]
   - 279.8803628401489 R[47,0] - 561.2671007743228 R[47,1]
   - 849.7184628604712 R[47,2] - 1131.105200674894 R[47,3]
   - 1406.49213925346 R[47,4] - 1687.878877063224 R[47,5]
   - 1961.399442504734 R[47,6] - 2237.79431714199 R[47,7] = 0
 Constr_Total_Demand_from_K_to_T_47: Total_Demand_from_k_to_T[47]
   - 2520.68742992137 R[47,0] - 2239.300691987195 R[47,1]
   - 1950.849329901047 R[47,2] - 1669.462592086624 R[47,3]
   - 1394.075653508059 R[47,4] - 1112.688915698295 R[47,5]
   - 839.168350256784 R[47,6] - 562.7734756195284 R[47,7] = 0
 Constr_Left_0_diff_aux_47: Sold_0[47] - Q0_var[47] + Left_0_aux_47 = 0
 Constr_Lost_0_diff_aux_47: - Total_Demand_Up_to_K_minus_1[47] + Q0_var[47]
   + Lost_0_aux_47 = 0
 Constr_Q1_plus_left_47: - Left_0[47] + Q1_plus_left[47] - Q1_var[47] = 0
 Constr_Left_1_diff_aux_47: Sold_1[47] - Q1_plus_left[47] + Left_1_aux_47
   = 0
 Constr_Lost_1_diff_aux_47: - Total_Demand_from_k_to_T[47]
   + Q1_plus_left[47] + Lost_1_aux_47 = 0
 Profit_Constraint_47: - 600 Sold_0[47] + 600 Lost_0[47] - 600 Sold_1[47]
   + 400 Left_1[47] + 600 Lost_1[47] + profits_vars[47] = 0
 R1488: - 16.86755507077842 alphas[0] - 270.071397465276 alphas[2]
   - alphas[3] + f_var[48] = 0
 Q0_upper_bound_48:
   - 2806.409713277834 Fraction_for_second_order_amount[48] + Q0_var[48]
   = 0
 tau_computation_48_0: - 16.86755507077842 betas[0,0]
   - 270.071397465276 betas[0,2] - betas[0,3] + tau[48,0] = 0
 tau_computation_48_1: - 16.86755507077842 betas[1,0]
   - 270.071397465276 betas[1,2] - betas[1,3] + tau[48,1] = 0
 tau_computation_48_2: - 16.86755507077842 betas[2,0]
   - 270.071397465276 betas[2,2] - betas[2,3] + tau[48,2] = 0
 tau_computation_48_3: - 16.86755507077842 betas[3,0]
   - 270.071397465276 betas[3,2] - betas[3,3] + tau[48,3] = 0
 tau_computation_48_4: - 16.86755507077842 betas[4,0]
   - 270.071397465276 betas[4,2] - betas[4,3] + tau[48,4] = 0
 tau_computation_48_5: - 16.86755507077842 betas[5,0]
   - 270.071397465276 betas[5,2] - betas[5,3] + tau[48,5] = 0
 tau_computation_48_6: - 16.86755507077842 betas[6,0]
   - 270.071397465276 betas[6,2] - betas[6,3] + tau[48,6] = 0
 tau_computation_48_7: - 16.86755507077842 betas[7,0]
   - 270.071397465276 betas[7,2] - betas[7,3] + tau[48,7] = 0
 tau_computation_48_8: - 16.86755507077842 betas[8,0]
   - 270.071397465276 betas[8,2] - betas[8,3] + tau[48,8] = 0
 tau_gap_48_0: tau[48,0] - 5e+06 R[48,0] - tau_star_48 <= -0.001
 tau_gap_48_1: tau[48,1] - 5e+06 R[48,1] - tau_star_48 <= -0.001
 tau_gap_48_2: tau[48,2] - 5e+06 R[48,2] - tau_star_48 <= -0.001
 tau_gap_48_3: tau[48,3] - 5e+06 R[48,3] - tau_star_48 <= -0.001
 tau_gap_48_4: tau[48,4] - 5e+06 R[48,4] - tau_star_48 <= -0.001
 tau_gap_48_5: tau[48,5] - 5e+06 R[48,5] - tau_star_48 <= -0.001
 tau_gap_48_6: tau[48,6] - 5e+06 R[48,6] - tau_star_48 <= -0.001
 tau_gap_48_7: tau[48,7] - 5e+06 R[48,7] - tau_star_48 <= -0.001
 tau_gap_48_8: tau[48,8] - 5e+06 R[48,8] - tau_star_48 <= -0.001
 one_R_48: R[48,0] + R[48,1] + R[48,2] + R[48,3] + R[48,4] + R[48,5]
   + R[48,6] + R[48,7] + R[48,8] = 1
 Define_Q_hat_48: - 2711.112172729813 R[48,0] - 2701.575209240521 R[48,1]
   - 2701.92329855982 R[48,2] - 2701.923297845311 R[48,3]
   - 2701.187087530443 R[48,4] - 2701.187087371436 R[48,5]
   - 2703.5676109942 R[48,6] - 2702.117746059326 R[48,7] + Q_hat[48] = 0
 Adjust_Q_hat_48: Q0_var[48] - Q_hat[48] + Q_hat_adjusted[48] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_48: Total_Demand_Up_to_K_minus_1[48]
   - 270.3437115138642 R[48,0] - 540.4151090677169 R[48,1]
   - 807.9777177099156 R[48,2] - 1078.049115168372 R[48,3]
   - 1348.998291672299 R[48,4] - 1619.069689137575 R[48,5]
   - 1893.217125542096 R[48,6] - 2161.974950456568 R[48,7] = 0
 Constr_Total_Demand_from_K_to_T_48: Total_Demand_from_k_to_T[48]
   - 2431.774033873256 R[48,0] - 2161.702636319403 R[48,1]
   - 1894.140027677204 R[48,2] - 1624.068630218748 R[48,3]
   - 1353.119453714821 R[48,4] - 1083.048056249545 R[48,5]
   - 808.9006198450238 R[48,6] - 540.142794930552 R[48,7] = 0
 Constr_Left_0_diff_aux_48: Sold_0[48] - Q0_var[48] + Left_0_aux_48 = 0
 Constr_Lost_0_diff_aux_48: - Total_Demand_Up_to_K_minus_1[48] + Q0_var[48]
   + Lost_0_aux_48 = 0
 Constr_Q1_plus_left_48: - Left_0[48] + Q1_plus_left[48] - Q1_var[48] = 0
 Constr_Left_1_diff_aux_48: Sold_1[48] - Q1_plus_left[48] + Left_1_aux_48
   = 0
 Constr_Lost_1_diff_aux_48: - Total_Demand_from_k_to_T[48]
   + Q1_plus_left[48] + Lost_1_aux_48 = 0
 Profit_Constraint_48: - 600 Sold_0[48] + 600 Lost_0[48] - 600 Sold_1[48]
   + 400 Left_1[48] + 600 Lost_1[48] + profits_vars[48] = 0
 R1519: - 17.57819173325626 alphas[0] - 270.8345878454355 alphas[2]
   - alphas[3] + f_var[49] = 0
 Q0_upper_bound_49:
   - 2806.409713277834 Fraction_for_second_order_amount[49] + Q0_var[49]
   = 0
 tau_computation_49_0: - 17.57819173325626 betas[0,0]
   - 270.8345878454355 betas[0,2] - betas[0,3] + tau[49,0] = 0
 tau_computation_49_1: - 17.57819173325626 betas[1,0]
   - 270.8345878454355 betas[1,2] - betas[1,3] + tau[49,1] = 0
 tau_computation_49_2: - 17.57819173325626 betas[2,0]
   - 270.8345878454355 betas[2,2] - betas[2,3] + tau[49,2] = 0
 tau_computation_49_3: - 17.57819173325626 betas[3,0]
   - 270.8345878454355 betas[3,2] - betas[3,3] + tau[49,3] = 0
 tau_computation_49_4: - 17.57819173325626 betas[4,0]
   - 270.8345878454355 betas[4,2] - betas[4,3] + tau[49,4] = 0
 tau_computation_49_5: - 17.57819173325626 betas[5,0]
   - 270.8345878454355 betas[5,2] - betas[5,3] + tau[49,5] = 0
 tau_computation_49_6: - 17.57819173325626 betas[6,0]
   - 270.8345878454355 betas[6,2] - betas[6,3] + tau[49,6] = 0
 tau_computation_49_7: - 17.57819173325626 betas[7,0]
   - 270.8345878454355 betas[7,2] - betas[7,3] + tau[49,7] = 0
 tau_computation_49_8: - 17.57819173325626 betas[8,0]
   - 270.8345878454355 betas[8,2] - betas[8,3] + tau[49,8] = 0
 tau_gap_49_0: tau[49,0] - 5e+06 R[49,0] - tau_star_49 <= -0.001
 tau_gap_49_1: tau[49,1] - 5e+06 R[49,1] - tau_star_49 <= -0.001
 tau_gap_49_2: tau[49,2] - 5e+06 R[49,2] - tau_star_49 <= -0.001
 tau_gap_49_3: tau[49,3] - 5e+06 R[49,3] - tau_star_49 <= -0.001
 tau_gap_49_4: tau[49,4] - 5e+06 R[49,4] - tau_star_49 <= -0.001
 tau_gap_49_5: tau[49,5] - 5e+06 R[49,5] - tau_star_49 <= -0.001
 tau_gap_49_6: tau[49,6] - 5e+06 R[49,6] - tau_star_49 <= -0.001
 tau_gap_49_7: tau[49,7] - 5e+06 R[49,7] - tau_star_49 <= -0.001
 tau_gap_49_8: tau[49,8] - 5e+06 R[49,8] - tau_star_49 <= -0.001
 one_R_49: R[49,0] + R[49,1] + R[49,2] + R[49,3] + R[49,4] + R[49,5]
   + R[49,6] + R[49,7] + R[49,8] = 1
 Define_Q_hat_49: - 2718.791173037975 R[49,0] - 2709.226504631546 R[49,1]
   - 2709.358208498198 R[49,2] - 2709.358207481437 R[49,3]
   - 2705.309305652885 R[49,4] - 2705.309305423546 R[49,5]
   - 2703.342984612246 R[49,6] - 2703.836847314492 R[49,7] + Q_hat[49] = 0
 Adjust_Q_hat_49: Q0_var[49] - Q_hat[49] + Q_hat_adjusted[49] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_49: Total_Demand_Up_to_K_minus_1[49]
   - 271.1680045547463 R[49,0] - 542.0025926107194 R[49,1]
   - 811.9894030001581 R[49,2] - 1082.823990845088 R[49,3]
   - 1351.301251077699 R[49,4] - 1622.135838923135 R[49,5]
   - 1891.234909457165 R[49,6] - 2162.167670861128 R[49,7] = 0
 Constr_Total_Demand_from_K_to_T_49: Total_Demand_from_k_to_T[49]
   - 2432.668841997253 R[49,0] - 2161.83425394128 R[49,1]
   - 1891.847443551841 R[49,2] - 1621.012855706911 R[49,3]
   - 1352.5355954743 R[49,4] - 1081.701007628864 R[49,5]
   - 812.6019370948344 R[49,6] - 541.669175690871 R[49,7] = 0
 Constr_Left_0_diff_aux_49: Sold_0[49] - Q0_var[49] + Left_0_aux_49 = 0
 Constr_Lost_0_diff_aux_49: - Total_Demand_Up_to_K_minus_1[49] + Q0_var[49]
   + Lost_0_aux_49 = 0
 Constr_Q1_plus_left_49: - Left_0[49] + Q1_plus_left[49] - Q1_var[49] = 0
 Constr_Left_1_diff_aux_49: Sold_1[49] - Q1_plus_left[49] + Left_1_aux_49
   = 0
 Constr_Lost_1_diff_aux_49: - Total_Demand_from_k_to_T[49]
   + Q1_plus_left[49] + Lost_1_aux_49 = 0
 Profit_Constraint_49: - 600 Sold_0[49] + 600 Lost_0[49] - 600 Sold_1[49]
   + 400 Left_1[49] + 600 Lost_1[49] + profits_vars[49] = 0
 tau_star_eq_0_0: R[0,0] = 1 -> tau[0,0] - tau_star_0 = 0
 tau_star_eq_0_1: R[0,1] = 1 -> tau[0,1] - tau_star_0 = 0
 tau_star_eq_0_2: R[0,2] = 1 -> tau[0,2] - tau_star_0 = 0
 tau_star_eq_0_3: R[0,3] = 1 -> tau[0,3] - tau_star_0 = 0
 tau_star_eq_0_4: R[0,4] = 1 -> tau[0,4] - tau_star_0 = 0
 tau_star_eq_0_5: R[0,5] = 1 -> tau[0,5] - tau_star_0 = 0
 tau_star_eq_0_6: R[0,6] = 1 -> tau[0,6] - tau_star_0 = 0
 tau_star_eq_0_7: R[0,7] = 1 -> tau[0,7] - tau_star_0 = 0
 tau_star_eq_0_8: R[0,8] = 1 -> tau[0,8] - tau_star_0 = 0
 tau_star_eq_1_0: R[1,0] = 1 -> tau[1,0] - tau_star_1 = 0
 tau_star_eq_1_1: R[1,1] = 1 -> tau[1,1] - tau_star_1 = 0
 tau_star_eq_1_2: R[1,2] = 1 -> tau[1,2] - tau_star_1 = 0
 tau_star_eq_1_3: R[1,3] = 1 -> tau[1,3] - tau_star_1 = 0
 tau_star_eq_1_4: R[1,4] = 1 -> tau[1,4] - tau_star_1 = 0
 tau_star_eq_1_5: R[1,5] = 1 -> tau[1,5] - tau_star_1 = 0
 tau_star_eq_1_6: R[1,6] = 1 -> tau[1,6] - tau_star_1 = 0
 tau_star_eq_1_7: R[1,7] = 1 -> tau[1,7] - tau_star_1 = 0
 tau_star_eq_1_8: R[1,8] = 1 -> tau[1,8] - tau_star_1 = 0
 tau_star_eq_2_0: R[2,0] = 1 -> tau[2,0] - tau_star_2 = 0
 tau_star_eq_2_1: R[2,1] = 1 -> tau[2,1] - tau_star_2 = 0
 tau_star_eq_2_2: R[2,2] = 1 -> tau[2,2] - tau_star_2 = 0
 tau_star_eq_2_3: R[2,3] = 1 -> tau[2,3] - tau_star_2 = 0
 tau_star_eq_2_4: R[2,4] = 1 -> tau[2,4] - tau_star_2 = 0
 tau_star_eq_2_5: R[2,5] = 1 -> tau[2,5] - tau_star_2 = 0
 tau_star_eq_2_6: R[2,6] = 1 -> tau[2,6] - tau_star_2 = 0
 tau_star_eq_2_7: R[2,7] = 1 -> tau[2,7] - tau_star_2 = 0
 tau_star_eq_2_8: R[2,8] = 1 -> tau[2,8] - tau_star_2 = 0
 tau_star_eq_3_0: R[3,0] = 1 -> tau[3,0] - tau_star_3 = 0
 tau_star_eq_3_1: R[3,1] = 1 -> tau[3,1] - tau_star_3 = 0
 tau_star_eq_3_2: R[3,2] = 1 -> tau[3,2] - tau_star_3 = 0
 tau_star_eq_3_3: R[3,3] = 1 -> tau[3,3] - tau_star_3 = 0
 tau_star_eq_3_4: R[3,4] = 1 -> tau[3,4] - tau_star_3 = 0
 tau_star_eq_3_5: R[3,5] = 1 -> tau[3,5] - tau_star_3 = 0
 tau_star_eq_3_6: R[3,6] = 1 -> tau[3,6] - tau_star_3 = 0
 tau_star_eq_3_7: R[3,7] = 1 -> tau[3,7] - tau_star_3 = 0
 tau_star_eq_3_8: R[3,8] = 1 -> tau[3,8] - tau_star_3 = 0
 tau_star_eq_4_0: R[4,0] = 1 -> tau[4,0] - tau_star_4 = 0
 tau_star_eq_4_1: R[4,1] = 1 -> tau[4,1] - tau_star_4 = 0
 tau_star_eq_4_2: R[4,2] = 1 -> tau[4,2] - tau_star_4 = 0
 tau_star_eq_4_3: R[4,3] = 1 -> tau[4,3] - tau_star_4 = 0
 tau_star_eq_4_4: R[4,4] = 1 -> tau[4,4] - tau_star_4 = 0
 tau_star_eq_4_5: R[4,5] = 1 -> tau[4,5] - tau_star_4 = 0
 tau_star_eq_4_6: R[4,6] = 1 -> tau[4,6] - tau_star_4 = 0
 tau_star_eq_4_7: R[4,7] = 1 -> tau[4,7] - tau_star_4 = 0
 tau_star_eq_4_8: R[4,8] = 1 -> tau[4,8] - tau_star_4 = 0
 tau_star_eq_5_0: R[5,0] = 1 -> tau[5,0] - tau_star_5 = 0
 tau_star_eq_5_1: R[5,1] = 1 -> tau[5,1] - tau_star_5 = 0
 tau_star_eq_5_2: R[5,2] = 1 -> tau[5,2] - tau_star_5 = 0
 tau_star_eq_5_3: R[5,3] = 1 -> tau[5,3] - tau_star_5 = 0
 tau_star_eq_5_4: R[5,4] = 1 -> tau[5,4] - tau_star_5 = 0
 tau_star_eq_5_5: R[5,5] = 1 -> tau[5,5] - tau_star_5 = 0
 tau_star_eq_5_6: R[5,6] = 1 -> tau[5,6] - tau_star_5 = 0
 tau_star_eq_5_7: R[5,7] = 1 -> tau[5,7] - tau_star_5 = 0
 tau_star_eq_5_8: R[5,8] = 1 -> tau[5,8] - tau_star_5 = 0
 tau_star_eq_6_0: R[6,0] = 1 -> tau[6,0] - tau_star_6 = 0
 tau_star_eq_6_1: R[6,1] = 1 -> tau[6,1] - tau_star_6 = 0
 tau_star_eq_6_2: R[6,2] = 1 -> tau[6,2] - tau_star_6 = 0
 tau_star_eq_6_3: R[6,3] = 1 -> tau[6,3] - tau_star_6 = 0
 tau_star_eq_6_4: R[6,4] = 1 -> tau[6,4] - tau_star_6 = 0
 tau_star_eq_6_5: R[6,5] = 1 -> tau[6,5] - tau_star_6 = 0
 tau_star_eq_6_6: R[6,6] = 1 -> tau[6,6] - tau_star_6 = 0
 tau_star_eq_6_7: R[6,7] = 1 -> tau[6,7] - tau_star_6 = 0
 tau_star_eq_6_8: R[6,8] = 1 -> tau[6,8] - tau_star_6 = 0
 tau_star_eq_7_0: R[7,0] = 1 -> tau[7,0] - tau_star_7 = 0
 tau_star_eq_7_1: R[7,1] = 1 -> tau[7,1] - tau_star_7 = 0
 tau_star_eq_7_2: R[7,2] = 1 -> tau[7,2] - tau_star_7 = 0
 tau_star_eq_7_3: R[7,3] = 1 -> tau[7,3] - tau_star_7 = 0
 tau_star_eq_7_4: R[7,4] = 1 -> tau[7,4] - tau_star_7 = 0
 tau_star_eq_7_5: R[7,5] = 1 -> tau[7,5] - tau_star_7 = 0
 tau_star_eq_7_6: R[7,6] = 1 -> tau[7,6] - tau_star_7 = 0
 tau_star_eq_7_7: R[7,7] = 1 -> tau[7,7] - tau_star_7 = 0
 tau_star_eq_7_8: R[7,8] = 1 -> tau[7,8] - tau_star_7 = 0
 tau_star_eq_8_0: R[8,0] = 1 -> tau[8,0] - tau_star_8 = 0
 tau_star_eq_8_1: R[8,1] = 1 -> tau[8,1] - tau_star_8 = 0
 tau_star_eq_8_2: R[8,2] = 1 -> tau[8,2] - tau_star_8 = 0
 tau_star_eq_8_3: R[8,3] = 1 -> tau[8,3] - tau_star_8 = 0
 tau_star_eq_8_4: R[8,4] = 1 -> tau[8,4] - tau_star_8 = 0
 tau_star_eq_8_5: R[8,5] = 1 -> tau[8,5] - tau_star_8 = 0
 tau_star_eq_8_6: R[8,6] = 1 -> tau[8,6] - tau_star_8 = 0
 tau_star_eq_8_7: R[8,7] = 1 -> tau[8,7] - tau_star_8 = 0
 tau_star_eq_8_8: R[8,8] = 1 -> tau[8,8] - tau_star_8 = 0
 tau_star_eq_9_0: R[9,0] = 1 -> tau[9,0] - tau_star_9 = 0
 tau_star_eq_9_1: R[9,1] = 1 -> tau[9,1] - tau_star_9 = 0
 tau_star_eq_9_2: R[9,2] = 1 -> tau[9,2] - tau_star_9 = 0
 tau_star_eq_9_3: R[9,3] = 1 -> tau[9,3] - tau_star_9 = 0
 tau_star_eq_9_4: R[9,4] = 1 -> tau[9,4] - tau_star_9 = 0
 tau_star_eq_9_5: R[9,5] = 1 -> tau[9,5] - tau_star_9 = 0
 tau_star_eq_9_6: R[9,6] = 1 -> tau[9,6] - tau_star_9 = 0
 tau_star_eq_9_7: R[9,7] = 1 -> tau[9,7] - tau_star_9 = 0
 tau_star_eq_9_8: R[9,8] = 1 -> tau[9,8] - tau_star_9 = 0
 tau_star_eq_10_0: R[10,0] = 1 -> tau[10,0] - tau_star_10 = 0
 tau_star_eq_10_1: R[10,1] = 1 -> tau[10,1] - tau_star_10 = 0
 tau_star_eq_10_2: R[10,2] = 1 -> tau[10,2] - tau_star_10 = 0
 tau_star_eq_10_3: R[10,3] = 1 -> tau[10,3] - tau_star_10 = 0
 tau_star_eq_10_4: R[10,4] = 1 -> tau[10,4] - tau_star_10 = 0
 tau_star_eq_10_5: R[10,5] = 1 -> tau[10,5] - tau_star_10 = 0
 tau_star_eq_10_6: R[10,6] = 1 -> tau[10,6] - tau_star_10 = 0
 tau_star_eq_10_7: R[10,7] = 1 -> tau[10,7] - tau_star_10 = 0
 tau_star_eq_10_8: R[10,8] = 1 -> tau[10,8] - tau_star_10 = 0
 tau_star_eq_11_0: R[11,0] = 1 -> tau[11,0] - tau_star_11 = 0
 tau_star_eq_11_1: R[11,1] = 1 -> tau[11,1] - tau_star_11 = 0
 tau_star_eq_11_2: R[11,2] = 1 -> tau[11,2] - tau_star_11 = 0
 tau_star_eq_11_3: R[11,3] = 1 -> tau[11,3] - tau_star_11 = 0
 tau_star_eq_11_4: R[11,4] = 1 -> tau[11,4] - tau_star_11 = 0
 tau_star_eq_11_5: R[11,5] = 1 -> tau[11,5] - tau_star_11 = 0
 tau_star_eq_11_6: R[11,6] = 1 -> tau[11,6] - tau_star_11 = 0
 tau_star_eq_11_7: R[11,7] = 1 -> tau[11,7] - tau_star_11 = 0
 tau_star_eq_11_8: R[11,8] = 1 -> tau[11,8] - tau_star_11 = 0
 tau_star_eq_12_0: R[12,0] = 1 -> tau[12,0] - tau_star_12 = 0
 tau_star_eq_12_1: R[12,1] = 1 -> tau[12,1] - tau_star_12 = 0
 tau_star_eq_12_2: R[12,2] = 1 -> tau[12,2] - tau_star_12 = 0
 tau_star_eq_12_3: R[12,3] = 1 -> tau[12,3] - tau_star_12 = 0
 tau_star_eq_12_4: R[12,4] = 1 -> tau[12,4] - tau_star_12 = 0
 tau_star_eq_12_5: R[12,5] = 1 -> tau[12,5] - tau_star_12 = 0
 tau_star_eq_12_6: R[12,6] = 1 -> tau[12,6] - tau_star_12 = 0
 tau_star_eq_12_7: R[12,7] = 1 -> tau[12,7] - tau_star_12 = 0
 tau_star_eq_12_8: R[12,8] = 1 -> tau[12,8] - tau_star_12 = 0
 tau_star_eq_13_0: R[13,0] = 1 -> tau[13,0] - tau_star_13 = 0
 tau_star_eq_13_1: R[13,1] = 1 -> tau[13,1] - tau_star_13 = 0
 tau_star_eq_13_2: R[13,2] = 1 -> tau[13,2] - tau_star_13 = 0
 tau_star_eq_13_3: R[13,3] = 1 -> tau[13,3] - tau_star_13 = 0
 tau_star_eq_13_4: R[13,4] = 1 -> tau[13,4] - tau_star_13 = 0
 tau_star_eq_13_5: R[13,5] = 1 -> tau[13,5] - tau_star_13 = 0
 tau_star_eq_13_6: R[13,6] = 1 -> tau[13,6] - tau_star_13 = 0
 tau_star_eq_13_7: R[13,7] = 1 -> tau[13,7] - tau_star_13 = 0
 tau_star_eq_13_8: R[13,8] = 1 -> tau[13,8] - tau_star_13 = 0
 tau_star_eq_14_0: R[14,0] = 1 -> tau[14,0] - tau_star_14 = 0
 tau_star_eq_14_1: R[14,1] = 1 -> tau[14,1] - tau_star_14 = 0
 tau_star_eq_14_2: R[14,2] = 1 -> tau[14,2] - tau_star_14 = 0
 tau_star_eq_14_3: R[14,3] = 1 -> tau[14,3] - tau_star_14 = 0
 tau_star_eq_14_4: R[14,4] = 1 -> tau[14,4] - tau_star_14 = 0
 tau_star_eq_14_5: R[14,5] = 1 -> tau[14,5] - tau_star_14 = 0
 tau_star_eq_14_6: R[14,6] = 1 -> tau[14,6] - tau_star_14 = 0
 tau_star_eq_14_7: R[14,7] = 1 -> tau[14,7] - tau_star_14 = 0
 tau_star_eq_14_8: R[14,8] = 1 -> tau[14,8] - tau_star_14 = 0
 tau_star_eq_15_0: R[15,0] = 1 -> tau[15,0] - tau_star_15 = 0
 tau_star_eq_15_1: R[15,1] = 1 -> tau[15,1] - tau_star_15 = 0
 tau_star_eq_15_2: R[15,2] = 1 -> tau[15,2] - tau_star_15 = 0
 tau_star_eq_15_3: R[15,3] = 1 -> tau[15,3] - tau_star_15 = 0
 tau_star_eq_15_4: R[15,4] = 1 -> tau[15,4] - tau_star_15 = 0
 tau_star_eq_15_5: R[15,5] = 1 -> tau[15,5] - tau_star_15 = 0
 tau_star_eq_15_6: R[15,6] = 1 -> tau[15,6] - tau_star_15 = 0
 tau_star_eq_15_7: R[15,7] = 1 -> tau[15,7] - tau_star_15 = 0
 tau_star_eq_15_8: R[15,8] = 1 -> tau[15,8] - tau_star_15 = 0
 tau_star_eq_16_0: R[16,0] = 1 -> tau[16,0] - tau_star_16 = 0
 tau_star_eq_16_1: R[16,1] = 1 -> tau[16,1] - tau_star_16 = 0
 tau_star_eq_16_2: R[16,2] = 1 -> tau[16,2] - tau_star_16 = 0
 tau_star_eq_16_3: R[16,3] = 1 -> tau[16,3] - tau_star_16 = 0
 tau_star_eq_16_4: R[16,4] = 1 -> tau[16,4] - tau_star_16 = 0
 tau_star_eq_16_5: R[16,5] = 1 -> tau[16,5] - tau_star_16 = 0
 tau_star_eq_16_6: R[16,6] = 1 -> tau[16,6] - tau_star_16 = 0
 tau_star_eq_16_7: R[16,7] = 1 -> tau[16,7] - tau_star_16 = 0
 tau_star_eq_16_8: R[16,8] = 1 -> tau[16,8] - tau_star_16 = 0
 tau_star_eq_17_0: R[17,0] = 1 -> tau[17,0] - tau_star_17 = 0
 tau_star_eq_17_1: R[17,1] = 1 -> tau[17,1] - tau_star_17 = 0
 tau_star_eq_17_2: R[17,2] = 1 -> tau[17,2] - tau_star_17 = 0
 tau_star_eq_17_3: R[17,3] = 1 -> tau[17,3] - tau_star_17 = 0
 tau_star_eq_17_4: R[17,4] = 1 -> tau[17,4] - tau_star_17 = 0
 tau_star_eq_17_5: R[17,5] = 1 -> tau[17,5] - tau_star_17 = 0
 tau_star_eq_17_6: R[17,6] = 1 -> tau[17,6] - tau_star_17 = 0
 tau_star_eq_17_7: R[17,7] = 1 -> tau[17,7] - tau_star_17 = 0
 tau_star_eq_17_8: R[17,8] = 1 -> tau[17,8] - tau_star_17 = 0
 tau_star_eq_18_0: R[18,0] = 1 -> tau[18,0] - tau_star_18 = 0
 tau_star_eq_18_1: R[18,1] = 1 -> tau[18,1] - tau_star_18 = 0
 tau_star_eq_18_2: R[18,2] = 1 -> tau[18,2] - tau_star_18 = 0
 tau_star_eq_18_3: R[18,3] = 1 -> tau[18,3] - tau_star_18 = 0
 tau_star_eq_18_4: R[18,4] = 1 -> tau[18,4] - tau_star_18 = 0
 tau_star_eq_18_5: R[18,5] = 1 -> tau[18,5] - tau_star_18 = 0
 tau_star_eq_18_6: R[18,6] = 1 -> tau[18,6] - tau_star_18 = 0
 tau_star_eq_18_7: R[18,7] = 1 -> tau[18,7] - tau_star_18 = 0
 tau_star_eq_18_8: R[18,8] = 1 -> tau[18,8] - tau_star_18 = 0
 tau_star_eq_19_0: R[19,0] = 1 -> tau[19,0] - tau_star_19 = 0
 tau_star_eq_19_1: R[19,1] = 1 -> tau[19,1] - tau_star_19 = 0
 tau_star_eq_19_2: R[19,2] = 1 -> tau[19,2] - tau_star_19 = 0
 tau_star_eq_19_3: R[19,3] = 1 -> tau[19,3] - tau_star_19 = 0
 tau_star_eq_19_4: R[19,4] = 1 -> tau[19,4] - tau_star_19 = 0
 tau_star_eq_19_5: R[19,5] = 1 -> tau[19,5] - tau_star_19 = 0
 tau_star_eq_19_6: R[19,6] = 1 -> tau[19,6] - tau_star_19 = 0
 tau_star_eq_19_7: R[19,7] = 1 -> tau[19,7] - tau_star_19 = 0
 tau_star_eq_19_8: R[19,8] = 1 -> tau[19,8] - tau_star_19 = 0
 tau_star_eq_20_0: R[20,0] = 1 -> tau[20,0] - tau_star_20 = 0
 tau_star_eq_20_1: R[20,1] = 1 -> tau[20,1] - tau_star_20 = 0
 tau_star_eq_20_2: R[20,2] = 1 -> tau[20,2] - tau_star_20 = 0
 tau_star_eq_20_3: R[20,3] = 1 -> tau[20,3] - tau_star_20 = 0
 tau_star_eq_20_4: R[20,4] = 1 -> tau[20,4] - tau_star_20 = 0
 tau_star_eq_20_5: R[20,5] = 1 -> tau[20,5] - tau_star_20 = 0
 tau_star_eq_20_6: R[20,6] = 1 -> tau[20,6] - tau_star_20 = 0
 tau_star_eq_20_7: R[20,7] = 1 -> tau[20,7] - tau_star_20 = 0
 tau_star_eq_20_8: R[20,8] = 1 -> tau[20,8] - tau_star_20 = 0
 tau_star_eq_21_0: R[21,0] = 1 -> tau[21,0] - tau_star_21 = 0
 tau_star_eq_21_1: R[21,1] = 1 -> tau[21,1] - tau_star_21 = 0
 tau_star_eq_21_2: R[21,2] = 1 -> tau[21,2] - tau_star_21 = 0
 tau_star_eq_21_3: R[21,3] = 1 -> tau[21,3] - tau_star_21 = 0
 tau_star_eq_21_4: R[21,4] = 1 -> tau[21,4] - tau_star_21 = 0
 tau_star_eq_21_5: R[21,5] = 1 -> tau[21,5] - tau_star_21 = 0
 tau_star_eq_21_6: R[21,6] = 1 -> tau[21,6] - tau_star_21 = 0
 tau_star_eq_21_7: R[21,7] = 1 -> tau[21,7] - tau_star_21 = 0
 tau_star_eq_21_8: R[21,8] = 1 -> tau[21,8] - tau_star_21 = 0
 tau_star_eq_22_0: R[22,0] = 1 -> tau[22,0] - tau_star_22 = 0
 tau_star_eq_22_1: R[22,1] = 1 -> tau[22,1] - tau_star_22 = 0
 tau_star_eq_22_2: R[22,2] = 1 -> tau[22,2] - tau_star_22 = 0
 tau_star_eq_22_3: R[22,3] = 1 -> tau[22,3] - tau_star_22 = 0
 tau_star_eq_22_4: R[22,4] = 1 -> tau[22,4] - tau_star_22 = 0
 tau_star_eq_22_5: R[22,5] = 1 -> tau[22,5] - tau_star_22 = 0
 tau_star_eq_22_6: R[22,6] = 1 -> tau[22,6] - tau_star_22 = 0
 tau_star_eq_22_7: R[22,7] = 1 -> tau[22,7] - tau_star_22 = 0
 tau_star_eq_22_8: R[22,8] = 1 -> tau[22,8] - tau_star_22 = 0
 tau_star_eq_23_0: R[23,0] = 1 -> tau[23,0] - tau_star_23 = 0
 tau_star_eq_23_1: R[23,1] = 1 -> tau[23,1] - tau_star_23 = 0
 tau_star_eq_23_2: R[23,2] = 1 -> tau[23,2] - tau_star_23 = 0
 tau_star_eq_23_3: R[23,3] = 1 -> tau[23,3] - tau_star_23 = 0
 tau_star_eq_23_4: R[23,4] = 1 -> tau[23,4] - tau_star_23 = 0
 tau_star_eq_23_5: R[23,5] = 1 -> tau[23,5] - tau_star_23 = 0
 tau_star_eq_23_6: R[23,6] = 1 -> tau[23,6] - tau_star_23 = 0
 tau_star_eq_23_7: R[23,7] = 1 -> tau[23,7] - tau_star_23 = 0
 tau_star_eq_23_8: R[23,8] = 1 -> tau[23,8] - tau_star_23 = 0
 tau_star_eq_24_0: R[24,0] = 1 -> tau[24,0] - tau_star_24 = 0
 tau_star_eq_24_1: R[24,1] = 1 -> tau[24,1] - tau_star_24 = 0
 tau_star_eq_24_2: R[24,2] = 1 -> tau[24,2] - tau_star_24 = 0
 tau_star_eq_24_3: R[24,3] = 1 -> tau[24,3] - tau_star_24 = 0
 tau_star_eq_24_4: R[24,4] = 1 -> tau[24,4] - tau_star_24 = 0
 tau_star_eq_24_5: R[24,5] = 1 -> tau[24,5] - tau_star_24 = 0
 tau_star_eq_24_6: R[24,6] = 1 -> tau[24,6] - tau_star_24 = 0
 tau_star_eq_24_7: R[24,7] = 1 -> tau[24,7] - tau_star_24 = 0
 tau_star_eq_24_8: R[24,8] = 1 -> tau[24,8] - tau_star_24 = 0
 tau_star_eq_25_0: R[25,0] = 1 -> tau[25,0] - tau_star_25 = 0
 tau_star_eq_25_1: R[25,1] = 1 -> tau[25,1] - tau_star_25 = 0
 tau_star_eq_25_2: R[25,2] = 1 -> tau[25,2] - tau_star_25 = 0
 tau_star_eq_25_3: R[25,3] = 1 -> tau[25,3] - tau_star_25 = 0
 tau_star_eq_25_4: R[25,4] = 1 -> tau[25,4] - tau_star_25 = 0
 tau_star_eq_25_5: R[25,5] = 1 -> tau[25,5] - tau_star_25 = 0
 tau_star_eq_25_6: R[25,6] = 1 -> tau[25,6] - tau_star_25 = 0
 tau_star_eq_25_7: R[25,7] = 1 -> tau[25,7] - tau_star_25 = 0
 tau_star_eq_25_8: R[25,8] = 1 -> tau[25,8] - tau_star_25 = 0
 tau_star_eq_26_0: R[26,0] = 1 -> tau[26,0] - tau_star_26 = 0
 tau_star_eq_26_1: R[26,1] = 1 -> tau[26,1] - tau_star_26 = 0
 tau_star_eq_26_2: R[26,2] = 1 -> tau[26,2] - tau_star_26 = 0
 tau_star_eq_26_3: R[26,3] = 1 -> tau[26,3] - tau_star_26 = 0
 tau_star_eq_26_4: R[26,4] = 1 -> tau[26,4] - tau_star_26 = 0
 tau_star_eq_26_5: R[26,5] = 1 -> tau[26,5] - tau_star_26 = 0
 tau_star_eq_26_6: R[26,6] = 1 -> tau[26,6] - tau_star_26 = 0
 tau_star_eq_26_7: R[26,7] = 1 -> tau[26,7] - tau_star_26 = 0
 tau_star_eq_26_8: R[26,8] = 1 -> tau[26,8] - tau_star_26 = 0
 tau_star_eq_27_0: R[27,0] = 1 -> tau[27,0] - tau_star_27 = 0
 tau_star_eq_27_1: R[27,1] = 1 -> tau[27,1] - tau_star_27 = 0
 tau_star_eq_27_2: R[27,2] = 1 -> tau[27,2] - tau_star_27 = 0
 tau_star_eq_27_3: R[27,3] = 1 -> tau[27,3] - tau_star_27 = 0
 tau_star_eq_27_4: R[27,4] = 1 -> tau[27,4] - tau_star_27 = 0
 tau_star_eq_27_5: R[27,5] = 1 -> tau[27,5] - tau_star_27 = 0
 tau_star_eq_27_6: R[27,6] = 1 -> tau[27,6] - tau_star_27 = 0
 tau_star_eq_27_7: R[27,7] = 1 -> tau[27,7] - tau_star_27 = 0
 tau_star_eq_27_8: R[27,8] = 1 -> tau[27,8] - tau_star_27 = 0
 tau_star_eq_28_0: R[28,0] = 1 -> tau[28,0] - tau_star_28 = 0
 tau_star_eq_28_1: R[28,1] = 1 -> tau[28,1] - tau_star_28 = 0
 tau_star_eq_28_2: R[28,2] = 1 -> tau[28,2] - tau_star_28 = 0
 tau_star_eq_28_3: R[28,3] = 1 -> tau[28,3] - tau_star_28 = 0
 tau_star_eq_28_4: R[28,4] = 1 -> tau[28,4] - tau_star_28 = 0
 tau_star_eq_28_5: R[28,5] = 1 -> tau[28,5] - tau_star_28 = 0
 tau_star_eq_28_6: R[28,6] = 1 -> tau[28,6] - tau_star_28 = 0
 tau_star_eq_28_7: R[28,7] = 1 -> tau[28,7] - tau_star_28 = 0
 tau_star_eq_28_8: R[28,8] = 1 -> tau[28,8] - tau_star_28 = 0
 tau_star_eq_29_0: R[29,0] = 1 -> tau[29,0] - tau_star_29 = 0
 tau_star_eq_29_1: R[29,1] = 1 -> tau[29,1] - tau_star_29 = 0
 tau_star_eq_29_2: R[29,2] = 1 -> tau[29,2] - tau_star_29 = 0
 tau_star_eq_29_3: R[29,3] = 1 -> tau[29,3] - tau_star_29 = 0
 tau_star_eq_29_4: R[29,4] = 1 -> tau[29,4] - tau_star_29 = 0
 tau_star_eq_29_5: R[29,5] = 1 -> tau[29,5] - tau_star_29 = 0
 tau_star_eq_29_6: R[29,6] = 1 -> tau[29,6] - tau_star_29 = 0
 tau_star_eq_29_7: R[29,7] = 1 -> tau[29,7] - tau_star_29 = 0
 tau_star_eq_29_8: R[29,8] = 1 -> tau[29,8] - tau_star_29 = 0
 tau_star_eq_30_0: R[30,0] = 1 -> tau[30,0] - tau_star_30 = 0
 tau_star_eq_30_1: R[30,1] = 1 -> tau[30,1] - tau_star_30 = 0
 tau_star_eq_30_2: R[30,2] = 1 -> tau[30,2] - tau_star_30 = 0
 tau_star_eq_30_3: R[30,3] = 1 -> tau[30,3] - tau_star_30 = 0
 tau_star_eq_30_4: R[30,4] = 1 -> tau[30,4] - tau_star_30 = 0
 tau_star_eq_30_5: R[30,5] = 1 -> tau[30,5] - tau_star_30 = 0
 tau_star_eq_30_6: R[30,6] = 1 -> tau[30,6] - tau_star_30 = 0
 tau_star_eq_30_7: R[30,7] = 1 -> tau[30,7] - tau_star_30 = 0
 tau_star_eq_30_8: R[30,8] = 1 -> tau[30,8] - tau_star_30 = 0
 tau_star_eq_31_0: R[31,0] = 1 -> tau[31,0] - tau_star_31 = 0
 tau_star_eq_31_1: R[31,1] = 1 -> tau[31,1] - tau_star_31 = 0
 tau_star_eq_31_2: R[31,2] = 1 -> tau[31,2] - tau_star_31 = 0
 tau_star_eq_31_3: R[31,3] = 1 -> tau[31,3] - tau_star_31 = 0
 tau_star_eq_31_4: R[31,4] = 1 -> tau[31,4] - tau_star_31 = 0
 tau_star_eq_31_5: R[31,5] = 1 -> tau[31,5] - tau_star_31 = 0
 tau_star_eq_31_6: R[31,6] = 1 -> tau[31,6] - tau_star_31 = 0
 tau_star_eq_31_7: R[31,7] = 1 -> tau[31,7] - tau_star_31 = 0
 tau_star_eq_31_8: R[31,8] = 1 -> tau[31,8] - tau_star_31 = 0
 tau_star_eq_32_0: R[32,0] = 1 -> tau[32,0] - tau_star_32 = 0
 tau_star_eq_32_1: R[32,1] = 1 -> tau[32,1] - tau_star_32 = 0
 tau_star_eq_32_2: R[32,2] = 1 -> tau[32,2] - tau_star_32 = 0
 tau_star_eq_32_3: R[32,3] = 1 -> tau[32,3] - tau_star_32 = 0
 tau_star_eq_32_4: R[32,4] = 1 -> tau[32,4] - tau_star_32 = 0
 tau_star_eq_32_5: R[32,5] = 1 -> tau[32,5] - tau_star_32 = 0
 tau_star_eq_32_6: R[32,6] = 1 -> tau[32,6] - tau_star_32 = 0
 tau_star_eq_32_7: R[32,7] = 1 -> tau[32,7] - tau_star_32 = 0
 tau_star_eq_32_8: R[32,8] = 1 -> tau[32,8] - tau_star_32 = 0
 tau_star_eq_33_0: R[33,0] = 1 -> tau[33,0] - tau_star_33 = 0
 tau_star_eq_33_1: R[33,1] = 1 -> tau[33,1] - tau_star_33 = 0
 tau_star_eq_33_2: R[33,2] = 1 -> tau[33,2] - tau_star_33 = 0
 tau_star_eq_33_3: R[33,3] = 1 -> tau[33,3] - tau_star_33 = 0
 tau_star_eq_33_4: R[33,4] = 1 -> tau[33,4] - tau_star_33 = 0
 tau_star_eq_33_5: R[33,5] = 1 -> tau[33,5] - tau_star_33 = 0
 tau_star_eq_33_6: R[33,6] = 1 -> tau[33,6] - tau_star_33 = 0
 tau_star_eq_33_7: R[33,7] = 1 -> tau[33,7] - tau_star_33 = 0
 tau_star_eq_33_8: R[33,8] = 1 -> tau[33,8] - tau_star_33 = 0
 tau_star_eq_34_0: R[34,0] = 1 -> tau[34,0] - tau_star_34 = 0
 tau_star_eq_34_1: R[34,1] = 1 -> tau[34,1] - tau_star_34 = 0
 tau_star_eq_34_2: R[34,2] = 1 -> tau[34,2] - tau_star_34 = 0
 tau_star_eq_34_3: R[34,3] = 1 -> tau[34,3] - tau_star_34 = 0
 tau_star_eq_34_4: R[34,4] = 1 -> tau[34,4] - tau_star_34 = 0
 tau_star_eq_34_5: R[34,5] = 1 -> tau[34,5] - tau_star_34 = 0
 tau_star_eq_34_6: R[34,6] = 1 -> tau[34,6] - tau_star_34 = 0
 tau_star_eq_34_7: R[34,7] = 1 -> tau[34,7] - tau_star_34 = 0
 tau_star_eq_34_8: R[34,8] = 1 -> tau[34,8] - tau_star_34 = 0
 tau_star_eq_35_0: R[35,0] = 1 -> tau[35,0] - tau_star_35 = 0
 tau_star_eq_35_1: R[35,1] = 1 -> tau[35,1] - tau_star_35 = 0
 tau_star_eq_35_2: R[35,2] = 1 -> tau[35,2] - tau_star_35 = 0
 tau_star_eq_35_3: R[35,3] = 1 -> tau[35,3] - tau_star_35 = 0
 tau_star_eq_35_4: R[35,4] = 1 -> tau[35,4] - tau_star_35 = 0
 tau_star_eq_35_5: R[35,5] = 1 -> tau[35,5] - tau_star_35 = 0
 tau_star_eq_35_6: R[35,6] = 1 -> tau[35,6] - tau_star_35 = 0
 tau_star_eq_35_7: R[35,7] = 1 -> tau[35,7] - tau_star_35 = 0
 tau_star_eq_35_8: R[35,8] = 1 -> tau[35,8] - tau_star_35 = 0
 tau_star_eq_36_0: R[36,0] = 1 -> tau[36,0] - tau_star_36 = 0
 tau_star_eq_36_1: R[36,1] = 1 -> tau[36,1] - tau_star_36 = 0
 tau_star_eq_36_2: R[36,2] = 1 -> tau[36,2] - tau_star_36 = 0
 tau_star_eq_36_3: R[36,3] = 1 -> tau[36,3] - tau_star_36 = 0
 tau_star_eq_36_4: R[36,4] = 1 -> tau[36,4] - tau_star_36 = 0
 tau_star_eq_36_5: R[36,5] = 1 -> tau[36,5] - tau_star_36 = 0
 tau_star_eq_36_6: R[36,6] = 1 -> tau[36,6] - tau_star_36 = 0
 tau_star_eq_36_7: R[36,7] = 1 -> tau[36,7] - tau_star_36 = 0
 tau_star_eq_36_8: R[36,8] = 1 -> tau[36,8] - tau_star_36 = 0
 tau_star_eq_37_0: R[37,0] = 1 -> tau[37,0] - tau_star_37 = 0
 tau_star_eq_37_1: R[37,1] = 1 -> tau[37,1] - tau_star_37 = 0
 tau_star_eq_37_2: R[37,2] = 1 -> tau[37,2] - tau_star_37 = 0
 tau_star_eq_37_3: R[37,3] = 1 -> tau[37,3] - tau_star_37 = 0
 tau_star_eq_37_4: R[37,4] = 1 -> tau[37,4] - tau_star_37 = 0
 tau_star_eq_37_5: R[37,5] = 1 -> tau[37,5] - tau_star_37 = 0
 tau_star_eq_37_6: R[37,6] = 1 -> tau[37,6] - tau_star_37 = 0
 tau_star_eq_37_7: R[37,7] = 1 -> tau[37,7] - tau_star_37 = 0
 tau_star_eq_37_8: R[37,8] = 1 -> tau[37,8] - tau_star_37 = 0
 tau_star_eq_38_0: R[38,0] = 1 -> tau[38,0] - tau_star_38 = 0
 tau_star_eq_38_1: R[38,1] = 1 -> tau[38,1] - tau_star_38 = 0
 tau_star_eq_38_2: R[38,2] = 1 -> tau[38,2] - tau_star_38 = 0
 tau_star_eq_38_3: R[38,3] = 1 -> tau[38,3] - tau_star_38 = 0
 tau_star_eq_38_4: R[38,4] = 1 -> tau[38,4] - tau_star_38 = 0
 tau_star_eq_38_5: R[38,5] = 1 -> tau[38,5] - tau_star_38 = 0
 tau_star_eq_38_6: R[38,6] = 1 -> tau[38,6] - tau_star_38 = 0
 tau_star_eq_38_7: R[38,7] = 1 -> tau[38,7] - tau_star_38 = 0
 tau_star_eq_38_8: R[38,8] = 1 -> tau[38,8] - tau_star_38 = 0
 tau_star_eq_39_0: R[39,0] = 1 -> tau[39,0] - tau_star_39 = 0
 tau_star_eq_39_1: R[39,1] = 1 -> tau[39,1] - tau_star_39 = 0
 tau_star_eq_39_2: R[39,2] = 1 -> tau[39,2] - tau_star_39 = 0
 tau_star_eq_39_3: R[39,3] = 1 -> tau[39,3] - tau_star_39 = 0
 tau_star_eq_39_4: R[39,4] = 1 -> tau[39,4] - tau_star_39 = 0
 tau_star_eq_39_5: R[39,5] = 1 -> tau[39,5] - tau_star_39 = 0
 tau_star_eq_39_6: R[39,6] = 1 -> tau[39,6] - tau_star_39 = 0
 tau_star_eq_39_7: R[39,7] = 1 -> tau[39,7] - tau_star_39 = 0
 tau_star_eq_39_8: R[39,8] = 1 -> tau[39,8] - tau_star_39 = 0
 tau_star_eq_40_0: R[40,0] = 1 -> tau[40,0] - tau_star_40 = 0
 tau_star_eq_40_1: R[40,1] = 1 -> tau[40,1] - tau_star_40 = 0
 tau_star_eq_40_2: R[40,2] = 1 -> tau[40,2] - tau_star_40 = 0
 tau_star_eq_40_3: R[40,3] = 1 -> tau[40,3] - tau_star_40 = 0
 tau_star_eq_40_4: R[40,4] = 1 -> tau[40,4] - tau_star_40 = 0
 tau_star_eq_40_5: R[40,5] = 1 -> tau[40,5] - tau_star_40 = 0
 tau_star_eq_40_6: R[40,6] = 1 -> tau[40,6] - tau_star_40 = 0
 tau_star_eq_40_7: R[40,7] = 1 -> tau[40,7] - tau_star_40 = 0
 tau_star_eq_40_8: R[40,8] = 1 -> tau[40,8] - tau_star_40 = 0
 tau_star_eq_41_0: R[41,0] = 1 -> tau[41,0] - tau_star_41 = 0
 tau_star_eq_41_1: R[41,1] = 1 -> tau[41,1] - tau_star_41 = 0
 tau_star_eq_41_2: R[41,2] = 1 -> tau[41,2] - tau_star_41 = 0
 tau_star_eq_41_3: R[41,3] = 1 -> tau[41,3] - tau_star_41 = 0
 tau_star_eq_41_4: R[41,4] = 1 -> tau[41,4] - tau_star_41 = 0
 tau_star_eq_41_5: R[41,5] = 1 -> tau[41,5] - tau_star_41 = 0
 tau_star_eq_41_6: R[41,6] = 1 -> tau[41,6] - tau_star_41 = 0
 tau_star_eq_41_7: R[41,7] = 1 -> tau[41,7] - tau_star_41 = 0
 tau_star_eq_41_8: R[41,8] = 1 -> tau[41,8] - tau_star_41 = 0
 tau_star_eq_42_0: R[42,0] = 1 -> tau[42,0] - tau_star_42 = 0
 tau_star_eq_42_1: R[42,1] = 1 -> tau[42,1] - tau_star_42 = 0
 tau_star_eq_42_2: R[42,2] = 1 -> tau[42,2] - tau_star_42 = 0
 tau_star_eq_42_3: R[42,3] = 1 -> tau[42,3] - tau_star_42 = 0
 tau_star_eq_42_4: R[42,4] = 1 -> tau[42,4] - tau_star_42 = 0
 tau_star_eq_42_5: R[42,5] = 1 -> tau[42,5] - tau_star_42 = 0
 tau_star_eq_42_6: R[42,6] = 1 -> tau[42,6] - tau_star_42 = 0
 tau_star_eq_42_7: R[42,7] = 1 -> tau[42,7] - tau_star_42 = 0
 tau_star_eq_42_8: R[42,8] = 1 -> tau[42,8] - tau_star_42 = 0
 tau_star_eq_43_0: R[43,0] = 1 -> tau[43,0] - tau_star_43 = 0
 tau_star_eq_43_1: R[43,1] = 1 -> tau[43,1] - tau_star_43 = 0
 tau_star_eq_43_2: R[43,2] = 1 -> tau[43,2] - tau_star_43 = 0
 tau_star_eq_43_3: R[43,3] = 1 -> tau[43,3] - tau_star_43 = 0
 tau_star_eq_43_4: R[43,4] = 1 -> tau[43,4] - tau_star_43 = 0
 tau_star_eq_43_5: R[43,5] = 1 -> tau[43,5] - tau_star_43 = 0
 tau_star_eq_43_6: R[43,6] = 1 -> tau[43,6] - tau_star_43 = 0
 tau_star_eq_43_7: R[43,7] = 1 -> tau[43,7] - tau_star_43 = 0
 tau_star_eq_43_8: R[43,8] = 1 -> tau[43,8] - tau_star_43 = 0
 tau_star_eq_44_0: R[44,0] = 1 -> tau[44,0] - tau_star_44 = 0
 tau_star_eq_44_1: R[44,1] = 1 -> tau[44,1] - tau_star_44 = 0
 tau_star_eq_44_2: R[44,2] = 1 -> tau[44,2] - tau_star_44 = 0
 tau_star_eq_44_3: R[44,3] = 1 -> tau[44,3] - tau_star_44 = 0
 tau_star_eq_44_4: R[44,4] = 1 -> tau[44,4] - tau_star_44 = 0
 tau_star_eq_44_5: R[44,5] = 1 -> tau[44,5] - tau_star_44 = 0
 tau_star_eq_44_6: R[44,6] = 1 -> tau[44,6] - tau_star_44 = 0
 tau_star_eq_44_7: R[44,7] = 1 -> tau[44,7] - tau_star_44 = 0
 tau_star_eq_44_8: R[44,8] = 1 -> tau[44,8] - tau_star_44 = 0
 tau_star_eq_45_0: R[45,0] = 1 -> tau[45,0] - tau_star_45 = 0
 tau_star_eq_45_1: R[45,1] = 1 -> tau[45,1] - tau_star_45 = 0
 tau_star_eq_45_2: R[45,2] = 1 -> tau[45,2] - tau_star_45 = 0
 tau_star_eq_45_3: R[45,3] = 1 -> tau[45,3] - tau_star_45 = 0
 tau_star_eq_45_4: R[45,4] = 1 -> tau[45,4] - tau_star_45 = 0
 tau_star_eq_45_5: R[45,5] = 1 -> tau[45,5] - tau_star_45 = 0
 tau_star_eq_45_6: R[45,6] = 1 -> tau[45,6] - tau_star_45 = 0
 tau_star_eq_45_7: R[45,7] = 1 -> tau[45,7] - tau_star_45 = 0
 tau_star_eq_45_8: R[45,8] = 1 -> tau[45,8] - tau_star_45 = 0
 tau_star_eq_46_0: R[46,0] = 1 -> tau[46,0] - tau_star_46 = 0
 tau_star_eq_46_1: R[46,1] = 1 -> tau[46,1] - tau_star_46 = 0
 tau_star_eq_46_2: R[46,2] = 1 -> tau[46,2] - tau_star_46 = 0
 tau_star_eq_46_3: R[46,3] = 1 -> tau[46,3] - tau_star_46 = 0
 tau_star_eq_46_4: R[46,4] = 1 -> tau[46,4] - tau_star_46 = 0
 tau_star_eq_46_5: R[46,5] = 1 -> tau[46,5] - tau_star_46 = 0
 tau_star_eq_46_6: R[46,6] = 1 -> tau[46,6] - tau_star_46 = 0
 tau_star_eq_46_7: R[46,7] = 1 -> tau[46,7] - tau_star_46 = 0
 tau_star_eq_46_8: R[46,8] = 1 -> tau[46,8] - tau_star_46 = 0
 tau_star_eq_47_0: R[47,0] = 1 -> tau[47,0] - tau_star_47 = 0
 tau_star_eq_47_1: R[47,1] = 1 -> tau[47,1] - tau_star_47 = 0
 tau_star_eq_47_2: R[47,2] = 1 -> tau[47,2] - tau_star_47 = 0
 tau_star_eq_47_3: R[47,3] = 1 -> tau[47,3] - tau_star_47 = 0
 tau_star_eq_47_4: R[47,4] = 1 -> tau[47,4] - tau_star_47 = 0
 tau_star_eq_47_5: R[47,5] = 1 -> tau[47,5] - tau_star_47 = 0
 tau_star_eq_47_6: R[47,6] = 1 -> tau[47,6] - tau_star_47 = 0
 tau_star_eq_47_7: R[47,7] = 1 -> tau[47,7] - tau_star_47 = 0
 tau_star_eq_47_8: R[47,8] = 1 -> tau[47,8] - tau_star_47 = 0
 tau_star_eq_48_0: R[48,0] = 1 -> tau[48,0] - tau_star_48 = 0
 tau_star_eq_48_1: R[48,1] = 1 -> tau[48,1] - tau_star_48 = 0
 tau_star_eq_48_2: R[48,2] = 1 -> tau[48,2] - tau_star_48 = 0
 tau_star_eq_48_3: R[48,3] = 1 -> tau[48,3] - tau_star_48 = 0
 tau_star_eq_48_4: R[48,4] = 1 -> tau[48,4] - tau_star_48 = 0
 tau_star_eq_48_5: R[48,5] = 1 -> tau[48,5] - tau_star_48 = 0
 tau_star_eq_48_6: R[48,6] = 1 -> tau[48,6] - tau_star_48 = 0
 tau_star_eq_48_7: R[48,7] = 1 -> tau[48,7] - tau_star_48 = 0
 tau_star_eq_48_8: R[48,8] = 1 -> tau[48,8] - tau_star_48 = 0
 tau_star_eq_49_0: R[49,0] = 1 -> tau[49,0] - tau_star_49 = 0
 tau_star_eq_49_1: R[49,1] = 1 -> tau[49,1] - tau_star_49 = 0
 tau_star_eq_49_2: R[49,2] = 1 -> tau[49,2] - tau_star_49 = 0
 tau_star_eq_49_3: R[49,3] = 1 -> tau[49,3] - tau_star_49 = 0
 tau_star_eq_49_4: R[49,4] = 1 -> tau[49,4] - tau_star_49 = 0
 tau_star_eq_49_5: R[49,5] = 1 -> tau[49,5] - tau_star_49 = 0
 tau_star_eq_49_6: R[49,6] = 1 -> tau[49,6] - tau_star_49 = 0
 tau_star_eq_49_7: R[49,7] = 1 -> tau[49,7] - tau_star_49 = 0
 tau_star_eq_49_8: R[49,8] = 1 -> tau[49,8] - tau_star_49 = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
 betas[0,0] free
 betas[0,1] free
 betas[0,2] free
 betas[0,3] free
 betas[1,0] free
 betas[1,1] free
 betas[1,2] free
 betas[1,3] free
 betas[2,0] free
 betas[2,1] free
 betas[2,2] free
 betas[2,3] free
 betas[3,0] free
 betas[3,1] free
 betas[3,2] free
 betas[3,3] free
 betas[4,0] free
 betas[4,1] free
 betas[4,2] free
 betas[4,3] free
 betas[5,0] free
 betas[5,1] free
 betas[5,2] free
 betas[5,3] free
 betas[6,0] free
 betas[6,1] free
 betas[6,2] free
 betas[6,3] free
 betas[7,0] free
 betas[7,1] free
 betas[7,2] free
 betas[7,3] free
 betas[8,0] free
 betas[8,1] free
 betas[8,2] free
 betas[8,3] free
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
 Q0_var[0] <= 2807.409713277834
 Q0_var[1] <= 2807.409713277834
 Q0_var[2] <= 2807.409713277834
 Q0_var[3] <= 2807.409713277834
 Q0_var[4] <= 2807.409713277834
 Q0_var[5] <= 2807.409713277834
 Q0_var[6] <= 2807.409713277834
 Q0_var[7] <= 2807.409713277834
 Q0_var[8] <= 2807.409713277834
 Q0_var[9] <= 2807.409713277834
 Q0_var[10] <= 2807.409713277834
 Q0_var[11] <= 2807.409713277834
 Q0_var[12] <= 2807.409713277834
 Q0_var[13] <= 2807.409713277834
 Q0_var[14] <= 2807.409713277834
 Q0_var[15] <= 2807.409713277834
 Q0_var[16] <= 2807.409713277834
 Q0_var[17] <= 2807.409713277834
 Q0_var[18] <= 2807.409713277834
 Q0_var[19] <= 2807.409713277834
 Q0_var[20] <= 2807.409713277834
 Q0_var[21] <= 2807.409713277834
 Q0_var[22] <= 2807.409713277834
 Q0_var[23] <= 2807.409713277834
 Q0_var[24] <= 2807.409713277834
 Q0_var[25] <= 2807.409713277834
 Q0_var[26] <= 2807.409713277834
 Q0_var[27] <= 2807.409713277834
 Q0_var[28] <= 2807.409713277834
 Q0_var[29] <= 2807.409713277834
 Q0_var[30] <= 2807.409713277834
 Q0_var[31] <= 2807.409713277834
 Q0_var[32] <= 2807.409713277834
 Q0_var[33] <= 2807.409713277834
 Q0_var[34] <= 2807.409713277834
 Q0_var[35] <= 2807.409713277834
 Q0_var[36] <= 2807.409713277834
 Q0_var[37] <= 2807.409713277834
 Q0_var[38] <= 2807.409713277834
 Q0_var[39] <= 2807.409713277834
 Q0_var[40] <= 2807.409713277834
 Q0_var[41] <= 2807.409713277834
 Q0_var[42] <= 2807.409713277834
 Q0_var[43] <= 2807.409713277834
 Q0_var[44] <= 2807.409713277834
 Q0_var[45] <= 2807.409713277834
 Q0_var[46] <= 2807.409713277834
 Q0_var[47] <= 2807.409713277834
 Q0_var[48] <= 2807.409713277834
 Q0_var[49] <= 2807.409713277834
 tau[0,0] free
 tau[0,1] free
 tau[0,2] free
 tau[0,3] free
 tau[0,4] free
 tau[0,5] free
 tau[0,6] free
 tau[0,7] free
 tau[0,8] free
 tau[1,0] free
 tau[1,1] free
 tau[1,2] free
 tau[1,3] free
 tau[1,4] free
 tau[1,5] free
 tau[1,6] free
 tau[1,7] free
 tau[1,8] free
 tau[2,0] free
 tau[2,1] free
 tau[2,2] free
 tau[2,3] free
 tau[2,4] free
 tau[2,5] free
 tau[2,6] free
 tau[2,7] free
 tau[2,8] free
 tau[3,0] free
 tau[3,1] free
 tau[3,2] free
 tau[3,3] free
 tau[3,4] free
 tau[3,5] free
 tau[3,6] free
 tau[3,7] free
 tau[3,8] free
 tau[4,0] free
 tau[4,1] free
 tau[4,2] free
 tau[4,3] free
 tau[4,4] free
 tau[4,5] free
 tau[4,6] free
 tau[4,7] free
 tau[4,8] free
 tau[5,0] free
 tau[5,1] free
 tau[5,2] free
 tau[5,3] free
 tau[5,4] free
 tau[5,5] free
 tau[5,6] free
 tau[5,7] free
 tau[5,8] free
 tau[6,0] free
 tau[6,1] free
 tau[6,2] free
 tau[6,3] free
 tau[6,4] free
 tau[6,5] free
 tau[6,6] free
 tau[6,7] free
 tau[6,8] free
 tau[7,0] free
 tau[7,1] free
 tau[7,2] free
 tau[7,3] free
 tau[7,4] free
 tau[7,5] free
 tau[7,6] free
 tau[7,7] free
 tau[7,8] free
 tau[8,0] free
 tau[8,1] free
 tau[8,2] free
 tau[8,3] free
 tau[8,4] free
 tau[8,5] free
 tau[8,6] free
 tau[8,7] free
 tau[8,8] free
 tau[9,0] free
 tau[9,1] free
 tau[9,2] free
 tau[9,3] free
 tau[9,4] free
 tau[9,5] free
 tau[9,6] free
 tau[9,7] free
 tau[9,8] free
 tau[10,0] free
 tau[10,1] free
 tau[10,2] free
 tau[10,3] free
 tau[10,4] free
 tau[10,5] free
 tau[10,6] free
 tau[10,7] free
 tau[10,8] free
 tau[11,0] free
 tau[11,1] free
 tau[11,2] free
 tau[11,3] free
 tau[11,4] free
 tau[11,5] free
 tau[11,6] free
 tau[11,7] free
 tau[11,8] free
 tau[12,0] free
 tau[12,1] free
 tau[12,2] free
 tau[12,3] free
 tau[12,4] free
 tau[12,5] free
 tau[12,6] free
 tau[12,7] free
 tau[12,8] free
 tau[13,0] free
 tau[13,1] free
 tau[13,2] free
 tau[13,3] free
 tau[13,4] free
 tau[13,5] free
 tau[13,6] free
 tau[13,7] free
 tau[13,8] free
 tau[14,0] free
 tau[14,1] free
 tau[14,2] free
 tau[14,3] free
 tau[14,4] free
 tau[14,5] free
 tau[14,6] free
 tau[14,7] free
 tau[14,8] free
 tau[15,0] free
 tau[15,1] free
 tau[15,2] free
 tau[15,3] free
 tau[15,4] free
 tau[15,5] free
 tau[15,6] free
 tau[15,7] free
 tau[15,8] free
 tau[16,0] free
 tau[16,1] free
 tau[16,2] free
 tau[16,3] free
 tau[16,4] free
 tau[16,5] free
 tau[16,6] free
 tau[16,7] free
 tau[16,8] free
 tau[17,0] free
 tau[17,1] free
 tau[17,2] free
 tau[17,3] free
 tau[17,4] free
 tau[17,5] free
 tau[17,6] free
 tau[17,7] free
 tau[17,8] free
 tau[18,0] free
 tau[18,1] free
 tau[18,2] free
 tau[18,3] free
 tau[18,4] free
 tau[18,5] free
 tau[18,6] free
 tau[18,7] free
 tau[18,8] free
 tau[19,0] free
 tau[19,1] free
 tau[19,2] free
 tau[19,3] free
 tau[19,4] free
 tau[19,5] free
 tau[19,6] free
 tau[19,7] free
 tau[19,8] free
 tau[20,0] free
 tau[20,1] free
 tau[20,2] free
 tau[20,3] free
 tau[20,4] free
 tau[20,5] free
 tau[20,6] free
 tau[20,7] free
 tau[20,8] free
 tau[21,0] free
 tau[21,1] free
 tau[21,2] free
 tau[21,3] free
 tau[21,4] free
 tau[21,5] free
 tau[21,6] free
 tau[21,7] free
 tau[21,8] free
 tau[22,0] free
 tau[22,1] free
 tau[22,2] free
 tau[22,3] free
 tau[22,4] free
 tau[22,5] free
 tau[22,6] free
 tau[22,7] free
 tau[22,8] free
 tau[23,0] free
 tau[23,1] free
 tau[23,2] free
 tau[23,3] free
 tau[23,4] free
 tau[23,5] free
 tau[23,6] free
 tau[23,7] free
 tau[23,8] free
 tau[24,0] free
 tau[24,1] free
 tau[24,2] free
 tau[24,3] free
 tau[24,4] free
 tau[24,5] free
 tau[24,6] free
 tau[24,7] free
 tau[24,8] free
 tau[25,0] free
 tau[25,1] free
 tau[25,2] free
 tau[25,3] free
 tau[25,4] free
 tau[25,5] free
 tau[25,6] free
 tau[25,7] free
 tau[25,8] free
 tau[26,0] free
 tau[26,1] free
 tau[26,2] free
 tau[26,3] free
 tau[26,4] free
 tau[26,5] free
 tau[26,6] free
 tau[26,7] free
 tau[26,8] free
 tau[27,0] free
 tau[27,1] free
 tau[27,2] free
 tau[27,3] free
 tau[27,4] free
 tau[27,5] free
 tau[27,6] free
 tau[27,7] free
 tau[27,8] free
 tau[28,0] free
 tau[28,1] free
 tau[28,2] free
 tau[28,3] free
 tau[28,4] free
 tau[28,5] free
 tau[28,6] free
 tau[28,7] free
 tau[28,8] free
 tau[29,0] free
 tau[29,1] free
 tau[29,2] free
 tau[29,3] free
 tau[29,4] free
 tau[29,5] free
 tau[29,6] free
 tau[29,7] free
 tau[29,8] free
 tau[30,0] free
 tau[30,1] free
 tau[30,2] free
 tau[30,3] free
 tau[30,4] free
 tau[30,5] free
 tau[30,6] free
 tau[30,7] free
 tau[30,8] free
 tau[31,0] free
 tau[31,1] free
 tau[31,2] free
 tau[31,3] free
 tau[31,4] free
 tau[31,5] free
 tau[31,6] free
 tau[31,7] free
 tau[31,8] free
 tau[32,0] free
 tau[32,1] free
 tau[32,2] free
 tau[32,3] free
 tau[32,4] free
 tau[32,5] free
 tau[32,6] free
 tau[32,7] free
 tau[32,8] free
 tau[33,0] free
 tau[33,1] free
 tau[33,2] free
 tau[33,3] free
 tau[33,4] free
 tau[33,5] free
 tau[33,6] free
 tau[33,7] free
 tau[33,8] free
 tau[34,0] free
 tau[34,1] free
 tau[34,2] free
 tau[34,3] free
 tau[34,4] free
 tau[34,5] free
 tau[34,6] free
 tau[34,7] free
 tau[34,8] free
 tau[35,0] free
 tau[35,1] free
 tau[35,2] free
 tau[35,3] free
 tau[35,4] free
 tau[35,5] free
 tau[35,6] free
 tau[35,7] free
 tau[35,8] free
 tau[36,0] free
 tau[36,1] free
 tau[36,2] free
 tau[36,3] free
 tau[36,4] free
 tau[36,5] free
 tau[36,6] free
 tau[36,7] free
 tau[36,8] free
 tau[37,0] free
 tau[37,1] free
 tau[37,2] free
 tau[37,3] free
 tau[37,4] free
 tau[37,5] free
 tau[37,6] free
 tau[37,7] free
 tau[37,8] free
 tau[38,0] free
 tau[38,1] free
 tau[38,2] free
 tau[38,3] free
 tau[38,4] free
 tau[38,5] free
 tau[38,6] free
 tau[38,7] free
 tau[38,8] free
 tau[39,0] free
 tau[39,1] free
 tau[39,2] free
 tau[39,3] free
 tau[39,4] free
 tau[39,5] free
 tau[39,6] free
 tau[39,7] free
 tau[39,8] free
 tau[40,0] free
 tau[40,1] free
 tau[40,2] free
 tau[40,3] free
 tau[40,4] free
 tau[40,5] free
 tau[40,6] free
 tau[40,7] free
 tau[40,8] free
 tau[41,0] free
 tau[41,1] free
 tau[41,2] free
 tau[41,3] free
 tau[41,4] free
 tau[41,5] free
 tau[41,6] free
 tau[41,7] free
 tau[41,8] free
 tau[42,0] free
 tau[42,1] free
 tau[42,2] free
 tau[42,3] free
 tau[42,4] free
 tau[42,5] free
 tau[42,6] free
 tau[42,7] free
 tau[42,8] free
 tau[43,0] free
 tau[43,1] free
 tau[43,2] free
 tau[43,3] free
 tau[43,4] free
 tau[43,5] free
 tau[43,6] free
 tau[43,7] free
 tau[43,8] free
 tau[44,0] free
 tau[44,1] free
 tau[44,2] free
 tau[44,3] free
 tau[44,4] free
 tau[44,5] free
 tau[44,6] free
 tau[44,7] free
 tau[44,8] free
 tau[45,0] free
 tau[45,1] free
 tau[45,2] free
 tau[45,3] free
 tau[45,4] free
 tau[45,5] free
 tau[45,6] free
 tau[45,7] free
 tau[45,8] free
 tau[46,0] free
 tau[46,1] free
 tau[46,2] free
 tau[46,3] free
 tau[46,4] free
 tau[46,5] free
 tau[46,6] free
 tau[46,7] free
 tau[46,8] free
 tau[47,0] free
 tau[47,1] free
 tau[47,2] free
 tau[47,3] free
 tau[47,4] free
 tau[47,5] free
 tau[47,6] free
 tau[47,7] free
 tau[47,8] free
 tau[48,0] free
 tau[48,1] free
 tau[48,2] free
 tau[48,3] free
 tau[48,4] free
 tau[48,5] free
 tau[48,6] free
 tau[48,7] free
 tau[48,8] free
 tau[49,0] free
 tau[49,1] free
 tau[49,2] free
 tau[49,3] free
 tau[49,4] free
 tau[49,5] free
 tau[49,6] free
 tau[49,7] free
 tau[49,8] free
 Q_hat_adjusted[0] free
 Q_hat_adjusted[1] free
 Q_hat_adjusted[2] free
 Q_hat_adjusted[3] free
 Q_hat_adjusted[4] free
 Q_hat_adjusted[5] free
 Q_hat_adjusted[6] free
 Q_hat_adjusted[7] free
 Q_hat_adjusted[8] free
 Q_hat_adjusted[9] free
 Q_hat_adjusted[10] free
 Q_hat_adjusted[11] free
 Q_hat_adjusted[12] free
 Q_hat_adjusted[13] free
 Q_hat_adjusted[14] free
 Q_hat_adjusted[15] free
 Q_hat_adjusted[16] free
 Q_hat_adjusted[17] free
 Q_hat_adjusted[18] free
 Q_hat_adjusted[19] free
 Q_hat_adjusted[20] free
 Q_hat_adjusted[21] free
 Q_hat_adjusted[22] free
 Q_hat_adjusted[23] free
 Q_hat_adjusted[24] free
 Q_hat_adjusted[25] free
 Q_hat_adjusted[26] free
 Q_hat_adjusted[27] free
 Q_hat_adjusted[28] free
 Q_hat_adjusted[29] free
 Q_hat_adjusted[30] free
 Q_hat_adjusted[31] free
 Q_hat_adjusted[32] free
 Q_hat_adjusted[33] free
 Q_hat_adjusted[34] free
 Q_hat_adjusted[35] free
 Q_hat_adjusted[36] free
 Q_hat_adjusted[37] free
 Q_hat_adjusted[38] free
 Q_hat_adjusted[39] free
 Q_hat_adjusted[40] free
 Q_hat_adjusted[41] free
 Q_hat_adjusted[42] free
 Q_hat_adjusted[43] free
 Q_hat_adjusted[44] free
 Q_hat_adjusted[45] free
 Q_hat_adjusted[46] free
 Q_hat_adjusted[47] free
 Q_hat_adjusted[48] free
 Q_hat_adjusted[49] free
 tau_star_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_1_aux_0 free
 Lost_1_aux_0 free
 tau_star_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_1_aux_1 free
 Lost_1_aux_1 free
 tau_star_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_1_aux_2 free
 Lost_1_aux_2 free
 tau_star_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_1_aux_3 free
 Lost_1_aux_3 free
 tau_star_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_1_aux_4 free
 Lost_1_aux_4 free
 tau_star_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_1_aux_5 free
 Lost_1_aux_5 free
 tau_star_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_1_aux_6 free
 Lost_1_aux_6 free
 tau_star_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_1_aux_7 free
 Lost_1_aux_7 free
 tau_star_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_1_aux_8 free
 Lost_1_aux_8 free
 tau_star_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_1_aux_9 free
 Lost_1_aux_9 free
 tau_star_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_1_aux_10 free
 Lost_1_aux_10 free
 tau_star_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_1_aux_11 free
 Lost_1_aux_11 free
 tau_star_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_1_aux_12 free
 Lost_1_aux_12 free
 tau_star_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_1_aux_13 free
 Lost_1_aux_13 free
 tau_star_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_1_aux_14 free
 Lost_1_aux_14 free
 tau_star_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_1_aux_15 free
 Lost_1_aux_15 free
 tau_star_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_1_aux_16 free
 Lost_1_aux_16 free
 tau_star_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_1_aux_17 free
 Lost_1_aux_17 free
 tau_star_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_1_aux_18 free
 Lost_1_aux_18 free
 tau_star_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_1_aux_19 free
 Lost_1_aux_19 free
 tau_star_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_1_aux_20 free
 Lost_1_aux_20 free
 tau_star_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_1_aux_21 free
 Lost_1_aux_21 free
 tau_star_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_1_aux_22 free
 Lost_1_aux_22 free
 tau_star_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_1_aux_23 free
 Lost_1_aux_23 free
 tau_star_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_1_aux_24 free
 Lost_1_aux_24 free
 tau_star_25 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Left_1_aux_25 free
 Lost_1_aux_25 free
 tau_star_26 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Left_1_aux_26 free
 Lost_1_aux_26 free
 tau_star_27 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Left_1_aux_27 free
 Lost_1_aux_27 free
 tau_star_28 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Left_1_aux_28 free
 Lost_1_aux_28 free
 tau_star_29 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Left_1_aux_29 free
 Lost_1_aux_29 free
 tau_star_30 free
 Left_0_aux_30 free
 Lost_0_aux_30 free
 Left_1_aux_30 free
 Lost_1_aux_30 free
 tau_star_31 free
 Left_0_aux_31 free
 Lost_0_aux_31 free
 Left_1_aux_31 free
 Lost_1_aux_31 free
 tau_star_32 free
 Left_0_aux_32 free
 Lost_0_aux_32 free
 Left_1_aux_32 free
 Lost_1_aux_32 free
 tau_star_33 free
 Left_0_aux_33 free
 Lost_0_aux_33 free
 Left_1_aux_33 free
 Lost_1_aux_33 free
 tau_star_34 free
 Left_0_aux_34 free
 Lost_0_aux_34 free
 Left_1_aux_34 free
 Lost_1_aux_34 free
 tau_star_35 free
 Left_0_aux_35 free
 Lost_0_aux_35 free
 Left_1_aux_35 free
 Lost_1_aux_35 free
 tau_star_36 free
 Left_0_aux_36 free
 Lost_0_aux_36 free
 Left_1_aux_36 free
 Lost_1_aux_36 free
 tau_star_37 free
 Left_0_aux_37 free
 Lost_0_aux_37 free
 Left_1_aux_37 free
 Lost_1_aux_37 free
 tau_star_38 free
 Left_0_aux_38 free
 Lost_0_aux_38 free
 Left_1_aux_38 free
 Lost_1_aux_38 free
 tau_star_39 free
 Left_0_aux_39 free
 Lost_0_aux_39 free
 Left_1_aux_39 free
 Lost_1_aux_39 free
 tau_star_40 free
 Left_0_aux_40 free
 Lost_0_aux_40 free
 Left_1_aux_40 free
 Lost_1_aux_40 free
 tau_star_41 free
 Left_0_aux_41 free
 Lost_0_aux_41 free
 Left_1_aux_41 free
 Lost_1_aux_41 free
 tau_star_42 free
 Left_0_aux_42 free
 Lost_0_aux_42 free
 Left_1_aux_42 free
 Lost_1_aux_42 free
 tau_star_43 free
 Left_0_aux_43 free
 Lost_0_aux_43 free
 Left_1_aux_43 free
 Lost_1_aux_43 free
 tau_star_44 free
 Left_0_aux_44 free
 Lost_0_aux_44 free
 Left_1_aux_44 free
 Lost_1_aux_44 free
 tau_star_45 free
 Left_0_aux_45 free
 Lost_0_aux_45 free
 Left_1_aux_45 free
 Lost_1_aux_45 free
 tau_star_46 free
 Left_0_aux_46 free
 Lost_0_aux_46 free
 Left_1_aux_46 free
 Lost_1_aux_46 free
 tau_star_47 free
 Left_0_aux_47 free
 Lost_0_aux_47 free
 Left_1_aux_47 free
 Lost_1_aux_47 free
 tau_star_48 free
 Left_0_aux_48 free
 Lost_0_aux_48 free
 Left_1_aux_48 free
 Lost_1_aux_48 free
 tau_star_49 free
 Left_0_aux_49 free
 Lost_0_aux_49 free
 Left_1_aux_49 free
 Lost_1_aux_49 free
Binaries
 R[0,0] R[0,1] R[0,2] R[0,3] R[0,4] R[0,5] R[0,6] R[0,7] R[0,8] R[1,0]
 R[1,1] R[1,2] R[1,3] R[1,4] R[1,5] R[1,6] R[1,7] R[1,8] R[2,0] R[2,1]
 R[2,2] R[2,3] R[2,4] R[2,5] R[2,6] R[2,7] R[2,8] R[3,0] R[3,1] R[3,2]
 R[3,3] R[3,4] R[3,5] R[3,6] R[3,7] R[3,8] R[4,0] R[4,1] R[4,2] R[4,3]
 R[4,4] R[4,5] R[4,6] R[4,7] R[4,8] R[5,0] R[5,1] R[5,2] R[5,3] R[5,4]
 R[5,5] R[5,6] R[5,7] R[5,8] R[6,0] R[6,1] R[6,2] R[6,3] R[6,4] R[6,5]
 R[6,6] R[6,7] R[6,8] R[7,0] R[7,1] R[7,2] R[7,3] R[7,4] R[7,5] R[7,6]
 R[7,7] R[7,8] R[8,0] R[8,1] R[8,2] R[8,3] R[8,4] R[8,5] R[8,6] R[8,7]
 R[8,8] R[9,0] R[9,1] R[9,2] R[9,3] R[9,4] R[9,5] R[9,6] R[9,7] R[9,8]
 R[10,0] R[10,1] R[10,2] R[10,3] R[10,4] R[10,5] R[10,6] R[10,7] R[10,8]
 R[11,0] R[11,1] R[11,2] R[11,3] R[11,4] R[11,5] R[11,6] R[11,7] R[11,8]
 R[12,0] R[12,1] R[12,2] R[12,3] R[12,4] R[12,5] R[12,6] R[12,7] R[12,8]
 R[13,0] R[13,1] R[13,2] R[13,3] R[13,4] R[13,5] R[13,6] R[13,7] R[13,8]
 R[14,0] R[14,1] R[14,2] R[14,3] R[14,4] R[14,5] R[14,6] R[14,7] R[14,8]
 R[15,0] R[15,1] R[15,2] R[15,3] R[15,4] R[15,5] R[15,6] R[15,7] R[15,8]
 R[16,0] R[16,1] R[16,2] R[16,3] R[16,4] R[16,5] R[16,6] R[16,7] R[16,8]
 R[17,0] R[17,1] R[17,2] R[17,3] R[17,4] R[17,5] R[17,6] R[17,7] R[17,8]
 R[18,0] R[18,1] R[18,2] R[18,3] R[18,4] R[18,5] R[18,6] R[18,7] R[18,8]
 R[19,0] R[19,1] R[19,2] R[19,3] R[19,4] R[19,5] R[19,6] R[19,7] R[19,8]
 R[20,0] R[20,1] R[20,2] R[20,3] R[20,4] R[20,5] R[20,6] R[20,7] R[20,8]
 R[21,0] R[21,1] R[21,2] R[21,3] R[21,4] R[21,5] R[21,6] R[21,7] R[21,8]
 R[22,0] R[22,1] R[22,2] R[22,3] R[22,4] R[22,5] R[22,6] R[22,7] R[22,8]
 R[23,0] R[23,1] R[23,2] R[23,3] R[23,4] R[23,5] R[23,6] R[23,7] R[23,8]
 R[24,0] R[24,1] R[24,2] R[24,3] R[24,4] R[24,5] R[24,6] R[24,7] R[24,8]
 R[25,0] R[25,1] R[25,2] R[25,3] R[25,4] R[25,5] R[25,6] R[25,7] R[25,8]
 R[26,0] R[26,1] R[26,2] R[26,3] R[26,4] R[26,5] R[26,6] R[26,7] R[26,8]
 R[27,0] R[27,1] R[27,2] R[27,3] R[27,4] R[27,5] R[27,6] R[27,7] R[27,8]
 R[28,0] R[28,1] R[28,2] R[28,3] R[28,4] R[28,5] R[28,6] R[28,7] R[28,8]
 R[29,0] R[29,1] R[29,2] R[29,3] R[29,4] R[29,5] R[29,6] R[29,7] R[29,8]
 R[30,0] R[30,1] R[30,2] R[30,3] R[30,4] R[30,5] R[30,6] R[30,7] R[30,8]
 R[31,0] R[31,1] R[31,2] R[31,3] R[31,4] R[31,5] R[31,6] R[31,7] R[31,8]
 R[32,0] R[32,1] R[32,2] R[32,3] R[32,4] R[32,5] R[32,6] R[32,7] R[32,8]
 R[33,0] R[33,1] R[33,2] R[33,3] R[33,4] R[33,5] R[33,6] R[33,7] R[33,8]
 R[34,0] R[34,1] R[34,2] R[34,3] R[34,4] R[34,5] R[34,6] R[34,7] R[34,8]
 R[35,0] R[35,1] R[35,2] R[35,3] R[35,4] R[35,5] R[35,6] R[35,7] R[35,8]
 R[36,0] R[36,1] R[36,2] R[36,3] R[36,4] R[36,5] R[36,6] R[36,7] R[36,8]
 R[37,0] R[37,1] R[37,2] R[37,3] R[37,4] R[37,5] R[37,6] R[37,7] R[37,8]
 R[38,0] R[38,1] R[38,2] R[38,3] R[38,4] R[38,5] R[38,6] R[38,7] R[38,8]
 R[39,0] R[39,1] R[39,2] R[39,3] R[39,4] R[39,5] R[39,6] R[39,7] R[39,8]
 R[40,0] R[40,1] R[40,2] R[40,3] R[40,4] R[40,5] R[40,6] R[40,7] R[40,8]
 R[41,0] R[41,1] R[41,2] R[41,3] R[41,4] R[41,5] R[41,6] R[41,7] R[41,8]
 R[42,0] R[42,1] R[42,2] R[42,3] R[42,4] R[42,5] R[42,6] R[42,7] R[42,8]
 R[43,0] R[43,1] R[43,2] R[43,3] R[43,4] R[43,5] R[43,6] R[43,7] R[43,8]
 R[44,0] R[44,1] R[44,2] R[44,3] R[44,4] R[44,5] R[44,6] R[44,7] R[44,8]
 R[45,0] R[45,1] R[45,2] R[45,3] R[45,4] R[45,5] R[45,6] R[45,7] R[45,8]
 R[46,0] R[46,1] R[46,2] R[46,3] R[46,4] R[46,5] R[46,6] R[46,7] R[46,8]
 R[47,0] R[47,1] R[47,2] R[47,3] R[47,4] R[47,5] R[47,6] R[47,7] R[47,8]
 R[48,0] R[48,1] R[48,2] R[48,3] R[48,4] R[48,5] R[48,6] R[48,7] R[48,8]
 R[49,0] R[49,1] R[49,2] R[49,3] R[49,4] R[49,5] R[49,6] R[49,7] R[49,8]
General Constraints
 logistic_constraint_0: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[0] = LOGISTIC ( f_var[0] )
 Max_Constraint_0: Q1_var[0] = MAX ( Q_hat_adjusted[0] , 0 )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Total_Demand_Up_to_K_minus_1[0] ,
   Q0_var[0] )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Total_Demand_from_k_to_T[0] ,
   Q1_plus_left[0] )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_1_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_1_aux_0 , 0 )
 logistic_constraint_1: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[1] = LOGISTIC ( f_var[1] )
 Max_Constraint_1: Q1_var[1] = MAX ( Q_hat_adjusted[1] , 0 )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Total_Demand_Up_to_K_minus_1[1] ,
   Q0_var[1] )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Total_Demand_from_k_to_T[1] ,
   Q1_plus_left[1] )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_1_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_1_aux_1 , 0 )
 logistic_constraint_2: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[2] = LOGISTIC ( f_var[2] )
 Max_Constraint_2: Q1_var[2] = MAX ( Q_hat_adjusted[2] , 0 )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Total_Demand_Up_to_K_minus_1[2] ,
   Q0_var[2] )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Total_Demand_from_k_to_T[2] ,
   Q1_plus_left[2] )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_1_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_1_aux_2 , 0 )
 logistic_constraint_3: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[3] = LOGISTIC ( f_var[3] )
 Max_Constraint_3: Q1_var[3] = MAX ( Q_hat_adjusted[3] , 0 )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Total_Demand_Up_to_K_minus_1[3] ,
   Q0_var[3] )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Total_Demand_from_k_to_T[3] ,
   Q1_plus_left[3] )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_1_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_1_aux_3 , 0 )
 logistic_constraint_4: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[4] = LOGISTIC ( f_var[4] )
 Max_Constraint_4: Q1_var[4] = MAX ( Q_hat_adjusted[4] , 0 )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Total_Demand_Up_to_K_minus_1[4] ,
   Q0_var[4] )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Total_Demand_from_k_to_T[4] ,
   Q1_plus_left[4] )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_1_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_1_aux_4 , 0 )
 logistic_constraint_5: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[5] = LOGISTIC ( f_var[5] )
 Max_Constraint_5: Q1_var[5] = MAX ( Q_hat_adjusted[5] , 0 )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Total_Demand_Up_to_K_minus_1[5] ,
   Q0_var[5] )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Total_Demand_from_k_to_T[5] ,
   Q1_plus_left[5] )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_1_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_1_aux_5 , 0 )
 logistic_constraint_6: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[6] = LOGISTIC ( f_var[6] )
 Max_Constraint_6: Q1_var[6] = MAX ( Q_hat_adjusted[6] , 0 )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Total_Demand_Up_to_K_minus_1[6] ,
   Q0_var[6] )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Total_Demand_from_k_to_T[6] ,
   Q1_plus_left[6] )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_1_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_1_aux_6 , 0 )
 logistic_constraint_7: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[7] = LOGISTIC ( f_var[7] )
 Max_Constraint_7: Q1_var[7] = MAX ( Q_hat_adjusted[7] , 0 )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Total_Demand_Up_to_K_minus_1[7] ,
   Q0_var[7] )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Total_Demand_from_k_to_T[7] ,
   Q1_plus_left[7] )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_1_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_1_aux_7 , 0 )
 logistic_constraint_8: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[8] = LOGISTIC ( f_var[8] )
 Max_Constraint_8: Q1_var[8] = MAX ( Q_hat_adjusted[8] , 0 )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Total_Demand_Up_to_K_minus_1[8] ,
   Q0_var[8] )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Total_Demand_from_k_to_T[8] ,
   Q1_plus_left[8] )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_1_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_1_aux_8 , 0 )
 logistic_constraint_9: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[9] = LOGISTIC ( f_var[9] )
 Max_Constraint_9: Q1_var[9] = MAX ( Q_hat_adjusted[9] , 0 )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Total_Demand_Up_to_K_minus_1[9] ,
   Q0_var[9] )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Total_Demand_from_k_to_T[9] ,
   Q1_plus_left[9] )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_1_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_1_aux_9 , 0 )
 logistic_constraint_10: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[10] = LOGISTIC ( f_var[10] )
 Max_Constraint_10: Q1_var[10] = MAX ( Q_hat_adjusted[10] , 0 )
 Constr_Sold_0_min_10: Sold_0[10] = MIN (
   Total_Demand_Up_to_K_minus_1[10] , Q0_var[10] )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Total_Demand_from_k_to_T[10] ,
   Q1_plus_left[10] )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_1_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_1_aux_10 , 0 )
 logistic_constraint_11: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[11] = LOGISTIC ( f_var[11] )
 Max_Constraint_11: Q1_var[11] = MAX ( Q_hat_adjusted[11] , 0 )
 Constr_Sold_0_min_11: Sold_0[11] = MIN (
   Total_Demand_Up_to_K_minus_1[11] , Q0_var[11] )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Total_Demand_from_k_to_T[11] ,
   Q1_plus_left[11] )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_1_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_1_aux_11 , 0 )
 logistic_constraint_12: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[12] = LOGISTIC ( f_var[12] )
 Max_Constraint_12: Q1_var[12] = MAX ( Q_hat_adjusted[12] , 0 )
 Constr_Sold_0_min_12: Sold_0[12] = MIN (
   Total_Demand_Up_to_K_minus_1[12] , Q0_var[12] )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Total_Demand_from_k_to_T[12] ,
   Q1_plus_left[12] )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_1_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_1_aux_12 , 0 )
 logistic_constraint_13: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[13] = LOGISTIC ( f_var[13] )
 Max_Constraint_13: Q1_var[13] = MAX ( Q_hat_adjusted[13] , 0 )
 Constr_Sold_0_min_13: Sold_0[13] = MIN (
   Total_Demand_Up_to_K_minus_1[13] , Q0_var[13] )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Total_Demand_from_k_to_T[13] ,
   Q1_plus_left[13] )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_1_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_1_aux_13 , 0 )
 logistic_constraint_14: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[14] = LOGISTIC ( f_var[14] )
 Max_Constraint_14: Q1_var[14] = MAX ( Q_hat_adjusted[14] , 0 )
 Constr_Sold_0_min_14: Sold_0[14] = MIN (
   Total_Demand_Up_to_K_minus_1[14] , Q0_var[14] )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Total_Demand_from_k_to_T[14] ,
   Q1_plus_left[14] )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_1_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_1_aux_14 , 0 )
 logistic_constraint_15: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[15] = LOGISTIC ( f_var[15] )
 Max_Constraint_15: Q1_var[15] = MAX ( Q_hat_adjusted[15] , 0 )
 Constr_Sold_0_min_15: Sold_0[15] = MIN (
   Total_Demand_Up_to_K_minus_1[15] , Q0_var[15] )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Total_Demand_from_k_to_T[15] ,
   Q1_plus_left[15] )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_1_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_1_aux_15 , 0 )
 logistic_constraint_16: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[16] = LOGISTIC ( f_var[16] )
 Max_Constraint_16: Q1_var[16] = MAX ( Q_hat_adjusted[16] , 0 )
 Constr_Sold_0_min_16: Sold_0[16] = MIN (
   Total_Demand_Up_to_K_minus_1[16] , Q0_var[16] )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Total_Demand_from_k_to_T[16] ,
   Q1_plus_left[16] )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_1_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_1_aux_16 , 0 )
 logistic_constraint_17: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[17] = LOGISTIC ( f_var[17] )
 Max_Constraint_17: Q1_var[17] = MAX ( Q_hat_adjusted[17] , 0 )
 Constr_Sold_0_min_17: Sold_0[17] = MIN (
   Total_Demand_Up_to_K_minus_1[17] , Q0_var[17] )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Total_Demand_from_k_to_T[17] ,
   Q1_plus_left[17] )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_1_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_1_aux_17 , 0 )
 logistic_constraint_18: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[18] = LOGISTIC ( f_var[18] )
 Max_Constraint_18: Q1_var[18] = MAX ( Q_hat_adjusted[18] , 0 )
 Constr_Sold_0_min_18: Sold_0[18] = MIN (
   Total_Demand_Up_to_K_minus_1[18] , Q0_var[18] )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Total_Demand_from_k_to_T[18] ,
   Q1_plus_left[18] )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_1_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_1_aux_18 , 0 )
 logistic_constraint_19: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[19] = LOGISTIC ( f_var[19] )
 Max_Constraint_19: Q1_var[19] = MAX ( Q_hat_adjusted[19] , 0 )
 Constr_Sold_0_min_19: Sold_0[19] = MIN (
   Total_Demand_Up_to_K_minus_1[19] , Q0_var[19] )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Total_Demand_from_k_to_T[19] ,
   Q1_plus_left[19] )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_1_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_1_aux_19 , 0 )
 logistic_constraint_20: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[20] = LOGISTIC ( f_var[20] )
 Max_Constraint_20: Q1_var[20] = MAX ( Q_hat_adjusted[20] , 0 )
 Constr_Sold_0_min_20: Sold_0[20] = MIN (
   Total_Demand_Up_to_K_minus_1[20] , Q0_var[20] )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Total_Demand_from_k_to_T[20] ,
   Q1_plus_left[20] )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_1_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_1_aux_20 , 0 )
 logistic_constraint_21: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[21] = LOGISTIC ( f_var[21] )
 Max_Constraint_21: Q1_var[21] = MAX ( Q_hat_adjusted[21] , 0 )
 Constr_Sold_0_min_21: Sold_0[21] = MIN (
   Total_Demand_Up_to_K_minus_1[21] , Q0_var[21] )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Total_Demand_from_k_to_T[21] ,
   Q1_plus_left[21] )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_1_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_1_aux_21 , 0 )
 logistic_constraint_22: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[22] = LOGISTIC ( f_var[22] )
 Max_Constraint_22: Q1_var[22] = MAX ( Q_hat_adjusted[22] , 0 )
 Constr_Sold_0_min_22: Sold_0[22] = MIN (
   Total_Demand_Up_to_K_minus_1[22] , Q0_var[22] )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Total_Demand_from_k_to_T[22] ,
   Q1_plus_left[22] )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_1_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_1_aux_22 , 0 )
 logistic_constraint_23: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[23] = LOGISTIC ( f_var[23] )
 Max_Constraint_23: Q1_var[23] = MAX ( Q_hat_adjusted[23] , 0 )
 Constr_Sold_0_min_23: Sold_0[23] = MIN (
   Total_Demand_Up_to_K_minus_1[23] , Q0_var[23] )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Total_Demand_from_k_to_T[23] ,
   Q1_plus_left[23] )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_1_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_1_aux_23 , 0 )
 logistic_constraint_24: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[24] = LOGISTIC ( f_var[24] )
 Max_Constraint_24: Q1_var[24] = MAX ( Q_hat_adjusted[24] , 0 )
 Constr_Sold_0_min_24: Sold_0[24] = MIN (
   Total_Demand_Up_to_K_minus_1[24] , Q0_var[24] )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Total_Demand_from_k_to_T[24] ,
   Q1_plus_left[24] )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_1_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_1_aux_24 , 0 )
 logistic_constraint_25: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[25] = LOGISTIC ( f_var[25] )
 Max_Constraint_25: Q1_var[25] = MAX ( Q_hat_adjusted[25] , 0 )
 Constr_Sold_0_min_25: Sold_0[25] = MIN (
   Total_Demand_Up_to_K_minus_1[25] , Q0_var[25] )
 Constr_Left_0_max_25: Left_0[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_0_max_25: Lost_0[25] = MAX ( Lost_0_aux_25 , 0 )
 Constr_Sold_1_min_25: Sold_1[25] = MIN ( Total_Demand_from_k_to_T[25] ,
   Q1_plus_left[25] )
 Constr_Left_1_max_25: Left_1[25] = MAX ( Left_1_aux_25 , 0 )
 Constr_Lost_1_max_25: Lost_1[25] = MAX ( Lost_1_aux_25 , 0 )
 logistic_constraint_26: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[26] = LOGISTIC ( f_var[26] )
 Max_Constraint_26: Q1_var[26] = MAX ( Q_hat_adjusted[26] , 0 )
 Constr_Sold_0_min_26: Sold_0[26] = MIN (
   Total_Demand_Up_to_K_minus_1[26] , Q0_var[26] )
 Constr_Left_0_max_26: Left_0[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_0_max_26: Lost_0[26] = MAX ( Lost_0_aux_26 , 0 )
 Constr_Sold_1_min_26: Sold_1[26] = MIN ( Total_Demand_from_k_to_T[26] ,
   Q1_plus_left[26] )
 Constr_Left_1_max_26: Left_1[26] = MAX ( Left_1_aux_26 , 0 )
 Constr_Lost_1_max_26: Lost_1[26] = MAX ( Lost_1_aux_26 , 0 )
 logistic_constraint_27: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[27] = LOGISTIC ( f_var[27] )
 Max_Constraint_27: Q1_var[27] = MAX ( Q_hat_adjusted[27] , 0 )
 Constr_Sold_0_min_27: Sold_0[27] = MIN (
   Total_Demand_Up_to_K_minus_1[27] , Q0_var[27] )
 Constr_Left_0_max_27: Left_0[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_0_max_27: Lost_0[27] = MAX ( Lost_0_aux_27 , 0 )
 Constr_Sold_1_min_27: Sold_1[27] = MIN ( Total_Demand_from_k_to_T[27] ,
   Q1_plus_left[27] )
 Constr_Left_1_max_27: Left_1[27] = MAX ( Left_1_aux_27 , 0 )
 Constr_Lost_1_max_27: Lost_1[27] = MAX ( Lost_1_aux_27 , 0 )
 logistic_constraint_28: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[28] = LOGISTIC ( f_var[28] )
 Max_Constraint_28: Q1_var[28] = MAX ( Q_hat_adjusted[28] , 0 )
 Constr_Sold_0_min_28: Sold_0[28] = MIN (
   Total_Demand_Up_to_K_minus_1[28] , Q0_var[28] )
 Constr_Left_0_max_28: Left_0[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_0_max_28: Lost_0[28] = MAX ( Lost_0_aux_28 , 0 )
 Constr_Sold_1_min_28: Sold_1[28] = MIN ( Total_Demand_from_k_to_T[28] ,
   Q1_plus_left[28] )
 Constr_Left_1_max_28: Left_1[28] = MAX ( Left_1_aux_28 , 0 )
 Constr_Lost_1_max_28: Lost_1[28] = MAX ( Lost_1_aux_28 , 0 )
 logistic_constraint_29: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[29] = LOGISTIC ( f_var[29] )
 Max_Constraint_29: Q1_var[29] = MAX ( Q_hat_adjusted[29] , 0 )
 Constr_Sold_0_min_29: Sold_0[29] = MIN (
   Total_Demand_Up_to_K_minus_1[29] , Q0_var[29] )
 Constr_Left_0_max_29: Left_0[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_0_max_29: Lost_0[29] = MAX ( Lost_0_aux_29 , 0 )
 Constr_Sold_1_min_29: Sold_1[29] = MIN ( Total_Demand_from_k_to_T[29] ,
   Q1_plus_left[29] )
 Constr_Left_1_max_29: Left_1[29] = MAX ( Left_1_aux_29 , 0 )
 Constr_Lost_1_max_29: Lost_1[29] = MAX ( Lost_1_aux_29 , 0 )
 logistic_constraint_30: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[30] = LOGISTIC ( f_var[30] )
 Max_Constraint_30: Q1_var[30] = MAX ( Q_hat_adjusted[30] , 0 )
 Constr_Sold_0_min_30: Sold_0[30] = MIN (
   Total_Demand_Up_to_K_minus_1[30] , Q0_var[30] )
 Constr_Left_0_max_30: Left_0[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_0_max_30: Lost_0[30] = MAX ( Lost_0_aux_30 , 0 )
 Constr_Sold_1_min_30: Sold_1[30] = MIN ( Total_Demand_from_k_to_T[30] ,
   Q1_plus_left[30] )
 Constr_Left_1_max_30: Left_1[30] = MAX ( Left_1_aux_30 , 0 )
 Constr_Lost_1_max_30: Lost_1[30] = MAX ( Lost_1_aux_30 , 0 )
 logistic_constraint_31: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[31] = LOGISTIC ( f_var[31] )
 Max_Constraint_31: Q1_var[31] = MAX ( Q_hat_adjusted[31] , 0 )
 Constr_Sold_0_min_31: Sold_0[31] = MIN (
   Total_Demand_Up_to_K_minus_1[31] , Q0_var[31] )
 Constr_Left_0_max_31: Left_0[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_0_max_31: Lost_0[31] = MAX ( Lost_0_aux_31 , 0 )
 Constr_Sold_1_min_31: Sold_1[31] = MIN ( Total_Demand_from_k_to_T[31] ,
   Q1_plus_left[31] )
 Constr_Left_1_max_31: Left_1[31] = MAX ( Left_1_aux_31 , 0 )
 Constr_Lost_1_max_31: Lost_1[31] = MAX ( Lost_1_aux_31 , 0 )
 logistic_constraint_32: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[32] = LOGISTIC ( f_var[32] )
 Max_Constraint_32: Q1_var[32] = MAX ( Q_hat_adjusted[32] , 0 )
 Constr_Sold_0_min_32: Sold_0[32] = MIN (
   Total_Demand_Up_to_K_minus_1[32] , Q0_var[32] )
 Constr_Left_0_max_32: Left_0[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_0_max_32: Lost_0[32] = MAX ( Lost_0_aux_32 , 0 )
 Constr_Sold_1_min_32: Sold_1[32] = MIN ( Total_Demand_from_k_to_T[32] ,
   Q1_plus_left[32] )
 Constr_Left_1_max_32: Left_1[32] = MAX ( Left_1_aux_32 , 0 )
 Constr_Lost_1_max_32: Lost_1[32] = MAX ( Lost_1_aux_32 , 0 )
 logistic_constraint_33: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[33] = LOGISTIC ( f_var[33] )
 Max_Constraint_33: Q1_var[33] = MAX ( Q_hat_adjusted[33] , 0 )
 Constr_Sold_0_min_33: Sold_0[33] = MIN (
   Total_Demand_Up_to_K_minus_1[33] , Q0_var[33] )
 Constr_Left_0_max_33: Left_0[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_0_max_33: Lost_0[33] = MAX ( Lost_0_aux_33 , 0 )
 Constr_Sold_1_min_33: Sold_1[33] = MIN ( Total_Demand_from_k_to_T[33] ,
   Q1_plus_left[33] )
 Constr_Left_1_max_33: Left_1[33] = MAX ( Left_1_aux_33 , 0 )
 Constr_Lost_1_max_33: Lost_1[33] = MAX ( Lost_1_aux_33 , 0 )
 logistic_constraint_34: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[34] = LOGISTIC ( f_var[34] )
 Max_Constraint_34: Q1_var[34] = MAX ( Q_hat_adjusted[34] , 0 )
 Constr_Sold_0_min_34: Sold_0[34] = MIN (
   Total_Demand_Up_to_K_minus_1[34] , Q0_var[34] )
 Constr_Left_0_max_34: Left_0[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_0_max_34: Lost_0[34] = MAX ( Lost_0_aux_34 , 0 )
 Constr_Sold_1_min_34: Sold_1[34] = MIN ( Total_Demand_from_k_to_T[34] ,
   Q1_plus_left[34] )
 Constr_Left_1_max_34: Left_1[34] = MAX ( Left_1_aux_34 , 0 )
 Constr_Lost_1_max_34: Lost_1[34] = MAX ( Lost_1_aux_34 , 0 )
 logistic_constraint_35: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[35] = LOGISTIC ( f_var[35] )
 Max_Constraint_35: Q1_var[35] = MAX ( Q_hat_adjusted[35] , 0 )
 Constr_Sold_0_min_35: Sold_0[35] = MIN (
   Total_Demand_Up_to_K_minus_1[35] , Q0_var[35] )
 Constr_Left_0_max_35: Left_0[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_0_max_35: Lost_0[35] = MAX ( Lost_0_aux_35 , 0 )
 Constr_Sold_1_min_35: Sold_1[35] = MIN ( Total_Demand_from_k_to_T[35] ,
   Q1_plus_left[35] )
 Constr_Left_1_max_35: Left_1[35] = MAX ( Left_1_aux_35 , 0 )
 Constr_Lost_1_max_35: Lost_1[35] = MAX ( Lost_1_aux_35 , 0 )
 logistic_constraint_36: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[36] = LOGISTIC ( f_var[36] )
 Max_Constraint_36: Q1_var[36] = MAX ( Q_hat_adjusted[36] , 0 )
 Constr_Sold_0_min_36: Sold_0[36] = MIN (
   Total_Demand_Up_to_K_minus_1[36] , Q0_var[36] )
 Constr_Left_0_max_36: Left_0[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_0_max_36: Lost_0[36] = MAX ( Lost_0_aux_36 , 0 )
 Constr_Sold_1_min_36: Sold_1[36] = MIN ( Total_Demand_from_k_to_T[36] ,
   Q1_plus_left[36] )
 Constr_Left_1_max_36: Left_1[36] = MAX ( Left_1_aux_36 , 0 )
 Constr_Lost_1_max_36: Lost_1[36] = MAX ( Lost_1_aux_36 , 0 )
 logistic_constraint_37: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[37] = LOGISTIC ( f_var[37] )
 Max_Constraint_37: Q1_var[37] = MAX ( Q_hat_adjusted[37] , 0 )
 Constr_Sold_0_min_37: Sold_0[37] = MIN (
   Total_Demand_Up_to_K_minus_1[37] , Q0_var[37] )
 Constr_Left_0_max_37: Left_0[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_0_max_37: Lost_0[37] = MAX ( Lost_0_aux_37 , 0 )
 Constr_Sold_1_min_37: Sold_1[37] = MIN ( Total_Demand_from_k_to_T[37] ,
   Q1_plus_left[37] )
 Constr_Left_1_max_37: Left_1[37] = MAX ( Left_1_aux_37 , 0 )
 Constr_Lost_1_max_37: Lost_1[37] = MAX ( Lost_1_aux_37 , 0 )
 logistic_constraint_38: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[38] = LOGISTIC ( f_var[38] )
 Max_Constraint_38: Q1_var[38] = MAX ( Q_hat_adjusted[38] , 0 )
 Constr_Sold_0_min_38: Sold_0[38] = MIN (
   Total_Demand_Up_to_K_minus_1[38] , Q0_var[38] )
 Constr_Left_0_max_38: Left_0[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_0_max_38: Lost_0[38] = MAX ( Lost_0_aux_38 , 0 )
 Constr_Sold_1_min_38: Sold_1[38] = MIN ( Total_Demand_from_k_to_T[38] ,
   Q1_plus_left[38] )
 Constr_Left_1_max_38: Left_1[38] = MAX ( Left_1_aux_38 , 0 )
 Constr_Lost_1_max_38: Lost_1[38] = MAX ( Lost_1_aux_38 , 0 )
 logistic_constraint_39: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[39] = LOGISTIC ( f_var[39] )
 Max_Constraint_39: Q1_var[39] = MAX ( Q_hat_adjusted[39] , 0 )
 Constr_Sold_0_min_39: Sold_0[39] = MIN (
   Total_Demand_Up_to_K_minus_1[39] , Q0_var[39] )
 Constr_Left_0_max_39: Left_0[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_0_max_39: Lost_0[39] = MAX ( Lost_0_aux_39 , 0 )
 Constr_Sold_1_min_39: Sold_1[39] = MIN ( Total_Demand_from_k_to_T[39] ,
   Q1_plus_left[39] )
 Constr_Left_1_max_39: Left_1[39] = MAX ( Left_1_aux_39 , 0 )
 Constr_Lost_1_max_39: Lost_1[39] = MAX ( Lost_1_aux_39 , 0 )
 logistic_constraint_40: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[40] = LOGISTIC ( f_var[40] )
 Max_Constraint_40: Q1_var[40] = MAX ( Q_hat_adjusted[40] , 0 )
 Constr_Sold_0_min_40: Sold_0[40] = MIN (
   Total_Demand_Up_to_K_minus_1[40] , Q0_var[40] )
 Constr_Left_0_max_40: Left_0[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_0_max_40: Lost_0[40] = MAX ( Lost_0_aux_40 , 0 )
 Constr_Sold_1_min_40: Sold_1[40] = MIN ( Total_Demand_from_k_to_T[40] ,
   Q1_plus_left[40] )
 Constr_Left_1_max_40: Left_1[40] = MAX ( Left_1_aux_40 , 0 )
 Constr_Lost_1_max_40: Lost_1[40] = MAX ( Lost_1_aux_40 , 0 )
 logistic_constraint_41: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[41] = LOGISTIC ( f_var[41] )
 Max_Constraint_41: Q1_var[41] = MAX ( Q_hat_adjusted[41] , 0 )
 Constr_Sold_0_min_41: Sold_0[41] = MIN (
   Total_Demand_Up_to_K_minus_1[41] , Q0_var[41] )
 Constr_Left_0_max_41: Left_0[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_0_max_41: Lost_0[41] = MAX ( Lost_0_aux_41 , 0 )
 Constr_Sold_1_min_41: Sold_1[41] = MIN ( Total_Demand_from_k_to_T[41] ,
   Q1_plus_left[41] )
 Constr_Left_1_max_41: Left_1[41] = MAX ( Left_1_aux_41 , 0 )
 Constr_Lost_1_max_41: Lost_1[41] = MAX ( Lost_1_aux_41 , 0 )
 logistic_constraint_42: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[42] = LOGISTIC ( f_var[42] )
 Max_Constraint_42: Q1_var[42] = MAX ( Q_hat_adjusted[42] , 0 )
 Constr_Sold_0_min_42: Sold_0[42] = MIN (
   Total_Demand_Up_to_K_minus_1[42] , Q0_var[42] )
 Constr_Left_0_max_42: Left_0[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_0_max_42: Lost_0[42] = MAX ( Lost_0_aux_42 , 0 )
 Constr_Sold_1_min_42: Sold_1[42] = MIN ( Total_Demand_from_k_to_T[42] ,
   Q1_plus_left[42] )
 Constr_Left_1_max_42: Left_1[42] = MAX ( Left_1_aux_42 , 0 )
 Constr_Lost_1_max_42: Lost_1[42] = MAX ( Lost_1_aux_42 , 0 )
 logistic_constraint_43: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[43] = LOGISTIC ( f_var[43] )
 Max_Constraint_43: Q1_var[43] = MAX ( Q_hat_adjusted[43] , 0 )
 Constr_Sold_0_min_43: Sold_0[43] = MIN (
   Total_Demand_Up_to_K_minus_1[43] , Q0_var[43] )
 Constr_Left_0_max_43: Left_0[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_0_max_43: Lost_0[43] = MAX ( Lost_0_aux_43 , 0 )
 Constr_Sold_1_min_43: Sold_1[43] = MIN ( Total_Demand_from_k_to_T[43] ,
   Q1_plus_left[43] )
 Constr_Left_1_max_43: Left_1[43] = MAX ( Left_1_aux_43 , 0 )
 Constr_Lost_1_max_43: Lost_1[43] = MAX ( Lost_1_aux_43 , 0 )
 logistic_constraint_44: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[44] = LOGISTIC ( f_var[44] )
 Max_Constraint_44: Q1_var[44] = MAX ( Q_hat_adjusted[44] , 0 )
 Constr_Sold_0_min_44: Sold_0[44] = MIN (
   Total_Demand_Up_to_K_minus_1[44] , Q0_var[44] )
 Constr_Left_0_max_44: Left_0[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_0_max_44: Lost_0[44] = MAX ( Lost_0_aux_44 , 0 )
 Constr_Sold_1_min_44: Sold_1[44] = MIN ( Total_Demand_from_k_to_T[44] ,
   Q1_plus_left[44] )
 Constr_Left_1_max_44: Left_1[44] = MAX ( Left_1_aux_44 , 0 )
 Constr_Lost_1_max_44: Lost_1[44] = MAX ( Lost_1_aux_44 , 0 )
 logistic_constraint_45: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[45] = LOGISTIC ( f_var[45] )
 Max_Constraint_45: Q1_var[45] = MAX ( Q_hat_adjusted[45] , 0 )
 Constr_Sold_0_min_45: Sold_0[45] = MIN (
   Total_Demand_Up_to_K_minus_1[45] , Q0_var[45] )
 Constr_Left_0_max_45: Left_0[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_0_max_45: Lost_0[45] = MAX ( Lost_0_aux_45 , 0 )
 Constr_Sold_1_min_45: Sold_1[45] = MIN ( Total_Demand_from_k_to_T[45] ,
   Q1_plus_left[45] )
 Constr_Left_1_max_45: Left_1[45] = MAX ( Left_1_aux_45 , 0 )
 Constr_Lost_1_max_45: Lost_1[45] = MAX ( Lost_1_aux_45 , 0 )
 logistic_constraint_46: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[46] = LOGISTIC ( f_var[46] )
 Max_Constraint_46: Q1_var[46] = MAX ( Q_hat_adjusted[46] , 0 )
 Constr_Sold_0_min_46: Sold_0[46] = MIN (
   Total_Demand_Up_to_K_minus_1[46] , Q0_var[46] )
 Constr_Left_0_max_46: Left_0[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_0_max_46: Lost_0[46] = MAX ( Lost_0_aux_46 , 0 )
 Constr_Sold_1_min_46: Sold_1[46] = MIN ( Total_Demand_from_k_to_T[46] ,
   Q1_plus_left[46] )
 Constr_Left_1_max_46: Left_1[46] = MAX ( Left_1_aux_46 , 0 )
 Constr_Lost_1_max_46: Lost_1[46] = MAX ( Lost_1_aux_46 , 0 )
 logistic_constraint_47: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[47] = LOGISTIC ( f_var[47] )
 Max_Constraint_47: Q1_var[47] = MAX ( Q_hat_adjusted[47] , 0 )
 Constr_Sold_0_min_47: Sold_0[47] = MIN (
   Total_Demand_Up_to_K_minus_1[47] , Q0_var[47] )
 Constr_Left_0_max_47: Left_0[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_0_max_47: Lost_0[47] = MAX ( Lost_0_aux_47 , 0 )
 Constr_Sold_1_min_47: Sold_1[47] = MIN ( Total_Demand_from_k_to_T[47] ,
   Q1_plus_left[47] )
 Constr_Left_1_max_47: Left_1[47] = MAX ( Left_1_aux_47 , 0 )
 Constr_Lost_1_max_47: Lost_1[47] = MAX ( Lost_1_aux_47 , 0 )
 logistic_constraint_48: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[48] = LOGISTIC ( f_var[48] )
 Max_Constraint_48: Q1_var[48] = MAX ( Q_hat_adjusted[48] , 0 )
 Constr_Sold_0_min_48: Sold_0[48] = MIN (
   Total_Demand_Up_to_K_minus_1[48] , Q0_var[48] )
 Constr_Left_0_max_48: Left_0[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_0_max_48: Lost_0[48] = MAX ( Lost_0_aux_48 , 0 )
 Constr_Sold_1_min_48: Sold_1[48] = MIN ( Total_Demand_from_k_to_T[48] ,
   Q1_plus_left[48] )
 Constr_Left_1_max_48: Left_1[48] = MAX ( Left_1_aux_48 , 0 )
 Constr_Lost_1_max_48: Lost_1[48] = MAX ( Lost_1_aux_48 , 0 )
 logistic_constraint_49: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[49] = LOGISTIC ( f_var[49] )
 Max_Constraint_49: Q1_var[49] = MAX ( Q_hat_adjusted[49] , 0 )
 Constr_Sold_0_min_49: Sold_0[49] = MIN (
   Total_Demand_Up_to_K_minus_1[49] , Q0_var[49] )
 Constr_Left_0_max_49: Left_0[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_0_max_49: Lost_0[49] = MAX ( Lost_0_aux_49 , 0 )
 Constr_Sold_1_min_49: Sold_1[49] = MIN ( Total_Demand_from_k_to_T[49] ,
   Q1_plus_left[49] )
 Constr_Left_1_max_49: Left_1[49] = MAX ( Left_1_aux_49 , 0 )
 Constr_Lost_1_max_49: Lost_1[49] = MAX ( Lost_1_aux_49 , 0 )
End
