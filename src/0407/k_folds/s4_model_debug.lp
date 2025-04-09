\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 Holding_Cost_0[0] + 0 Holding_Cost_0[1] + 0 Holding_Cost_0[2]
   + 0 Holding_Cost_0[3] + 0 Holding_Cost_0[4] + 0 Holding_Cost_0[5]
   + 0 Holding_Cost_0[6] + 0 Holding_Cost_0[7] + 0 Holding_Cost_0[8]
   + 0 Holding_Cost_0[9] + 0 Holding_Cost_0[10] + 0 Holding_Cost_0[11]
   + 0 Holding_Cost_0[12] + 0 Holding_Cost_0[13] + 0 Holding_Cost_0[14]
   + 0 Holding_Cost_1[0] + 0 Holding_Cost_1[1] + 0 Holding_Cost_1[2]
   + 0 Holding_Cost_1[3] + 0 Holding_Cost_1[4] + 0 Holding_Cost_1[5]
   + 0 Holding_Cost_1[6] + 0 Holding_Cost_1[7] + 0 Holding_Cost_1[8]
   + 0 Holding_Cost_1[9] + 0 Holding_Cost_1[10] + 0 Holding_Cost_1[11]
   + 0 Holding_Cost_1[12] + 0 Holding_Cost_1[13] + 0 Holding_Cost_1[14]
   + profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14] + 0 r[0,0] + 0 r[0,1] + 0 r[0,2] + 0 r[0,3]
   + 0 r[0,4] + 0 r[0,5] + 0 r[0,6] + 0 r[0,7] + 0 r[1,0] + 0 r[1,1]
   + 0 r[1,2] + 0 r[1,3] + 0 r[1,4] + 0 r[1,5] + 0 r[1,6] + 0 r[1,7]
   + 0 r[2,0] + 0 r[2,1] + 0 r[2,2] + 0 r[2,3] + 0 r[2,4] + 0 r[2,5]
   + 0 r[2,6] + 0 r[2,7] + 0 r[3,0] + 0 r[3,1] + 0 r[3,2] + 0 r[3,3]
   + 0 r[3,4] + 0 r[3,5] + 0 r[3,6] + 0 r[3,7] + 0 r[4,0] + 0 r[4,1]
   + 0 r[4,2] + 0 r[4,3] + 0 r[4,4] + 0 r[4,5] + 0 r[4,6] + 0 r[4,7]
   + 0 r[5,0] + 0 r[5,1] + 0 r[5,2] + 0 r[5,3] + 0 r[5,4] + 0 r[5,5]
   + 0 r[5,6] + 0 r[5,7] + 0 r[6,0] + 0 r[6,1] + 0 r[6,2] + 0 r[6,3]
   + 0 r[6,4] + 0 r[6,5] + 0 r[6,6] + 0 r[6,7] + 0 r[7,0] + 0 r[7,1]
   + 0 r[7,2] + 0 r[7,3] + 0 r[7,4] + 0 r[7,5] + 0 r[7,6] + 0 r[7,7]
   + 0 r[8,0] + 0 r[8,1] + 0 r[8,2] + 0 r[8,3] + 0 r[8,4] + 0 r[8,5]
   + 0 r[8,6] + 0 r[8,7] + 0 r[9,0] + 0 r[9,1] + 0 r[9,2] + 0 r[9,3]
   + 0 r[9,4] + 0 r[9,5] + 0 r[9,6] + 0 r[9,7] + 0 r[10,0] + 0 r[10,1]
   + 0 r[10,2] + 0 r[10,3] + 0 r[10,4] + 0 r[10,5] + 0 r[10,6] + 0 r[10,7]
   + 0 r[11,0] + 0 r[11,1] + 0 r[11,2] + 0 r[11,3] + 0 r[11,4] + 0 r[11,5]
   + 0 r[11,6] + 0 r[11,7] + 0 r[12,0] + 0 r[12,1] + 0 r[12,2] + 0 r[12,3]
   + 0 r[12,4] + 0 r[12,5] + 0 r[12,6] + 0 r[12,7] + 0 r[13,0] + 0 r[13,1]
   + 0 r[13,2] + 0 r[13,3] + 0 r[13,4] + 0 r[13,5] + 0 r[13,6] + 0 r[13,7]
   + 0 r[14,0] + 0 r[14,1] + 0 r[14,2] + 0 r[14,3] + 0 r[14,4] + 0 r[14,5]
   + 0 r[14,6] + 0 r[14,7]
Subject To
 R0: - 100 alphas[0] - alphas[1] - 2.078375584508641 alphas[2] - alphas[3]
   + f_var[0] = 0
 Q0_upper_bound_0: - 2439.950183572415 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_0_0: - 100 betas[0,0] - betas[0,1]
   - 2.078375584508641 betas[0,2] - betas[0,3] + tau[0,0] = 0
 tau_computation_0_1: - 100 betas[1,0] - betas[1,1]
   - 2.078375584508641 betas[1,2] - betas[1,3] + tau[0,1] = 0
 tau_computation_0_2: - 100 betas[2,0] - betas[2,1]
   - 2.078375584508641 betas[2,2] - betas[2,3] + tau[0,2] = 0
 tau_computation_0_3: - 100 betas[3,0] - betas[3,1]
   - 2.078375584508641 betas[3,2] - betas[3,3] + tau[0,3] = 0
 tau_computation_0_4: - 100 betas[4,0] - betas[4,1]
   - 2.078375584508641 betas[4,2] - betas[4,3] + tau[0,4] = 0
 tau_computation_0_5: - 100 betas[5,0] - betas[5,1]
   - 2.078375584508641 betas[5,2] - betas[5,3] + tau[0,5] = 0
 tau_computation_0_6: - 100 betas[6,0] - betas[6,1]
   - 2.078375584508641 betas[6,2] - betas[6,3] + tau[0,6] = 0
 tau_computation_0_7: - 100 betas[7,0] - betas[7,1]
   - 2.078375584508641 betas[7,2] - betas[7,3] + tau[0,7] = 0
 tau_gap_0_0: tau[0,0] - 5e+06 R[0,0] - tau_star_0 <= -0.001
 tau_gap_0_1: tau[0,1] - 5e+06 R[0,1] - tau_star_0 <= -0.001
 tau_gap_0_2: tau[0,2] - 5e+06 R[0,2] - tau_star_0 <= -0.001
 tau_gap_0_3: tau[0,3] - 5e+06 R[0,3] - tau_star_0 <= -0.001
 tau_gap_0_4: tau[0,4] - 5e+06 R[0,4] - tau_star_0 <= -0.001
 tau_gap_0_5: tau[0,5] - 5e+06 R[0,5] - tau_star_0 <= -0.001
 tau_gap_0_6: tau[0,6] - 5e+06 R[0,6] - tau_star_0 <= -0.001
 tau_gap_0_7: tau[0,7] - 5e+06 R[0,7] - tau_star_0 <= -0.001
 one_R_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4] + R[0,5] + R[0,6]
   + R[0,7] = 1
 Define_Q_hat_0: - 1327.5560908169 R[0,0] - 1313.863759585685 R[0,1]
   - 1323.75355486305 R[0,2] - 1341.450026702626 R[0,3]
   - 1340.901500594861 R[0,4] - 1336.798594300918 R[0,5]
   - 1333.242990663352 R[0,6] - 1335.642211179191 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 76.72600695350441 R[0,0] - 162.0189159237364 R[0,1]
   - 295.5185443832647 R[0,2] - 438.2053712768976 R[0,3]
   - 590.1952777831037 R[0,4] - 738.2949272166156 R[0,5]
   - 886.2119433677973 R[0,6] - 1039.326879148593 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1256.719603642134 R[0,0] - 1171.426694671901 R[0,1]
   - 1037.927066212373 R[0,2] - 895.2402393187404 R[0,3]
   - 743.2503328125342 R[0,4] - 595.1506833790222 R[0,5]
   - 447.2336672278405 R[0,6] - 294.1187314470454 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 600 Sold_0[0] + 400 Left_0[0] + 600 Lost_0[0]
   - 600 Sold_1[0] + 400 Left_1[0] + 600 Lost_1[0] + profits_vars[0] = 0
 R29: - 100 alphas[0] - alphas[1] - 2.565613925152819 alphas[2] - alphas[3]
   + f_var[1] = 0
 Q0_upper_bound_1: - 2439.950183572415 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_1_0: - 100 betas[0,0] - betas[0,1]
   - 2.565613925152819 betas[0,2] - betas[0,3] + tau[1,0] = 0
 tau_computation_1_1: - 100 betas[1,0] - betas[1,1]
   - 2.565613925152819 betas[1,2] - betas[1,3] + tau[1,1] = 0
 tau_computation_1_2: - 100 betas[2,0] - betas[2,1]
   - 2.565613925152819 betas[2,2] - betas[2,3] + tau[1,2] = 0
 tau_computation_1_3: - 100 betas[3,0] - betas[3,1]
   - 2.565613925152819 betas[3,2] - betas[3,3] + tau[1,3] = 0
 tau_computation_1_4: - 100 betas[4,0] - betas[4,1]
   - 2.565613925152819 betas[4,2] - betas[4,3] + tau[1,4] = 0
 tau_computation_1_5: - 100 betas[5,0] - betas[5,1]
   - 2.565613925152819 betas[5,2] - betas[5,3] + tau[1,5] = 0
 tau_computation_1_6: - 100 betas[6,0] - betas[6,1]
   - 2.565613925152819 betas[6,2] - betas[6,3] + tau[1,6] = 0
 tau_computation_1_7: - 100 betas[7,0] - betas[7,1]
   - 2.565613925152819 betas[7,2] - betas[7,3] + tau[1,7] = 0
 tau_gap_1_0: tau[1,0] - 5e+06 R[1,0] - tau_star_1 <= -0.001
 tau_gap_1_1: tau[1,1] - 5e+06 R[1,1] - tau_star_1 <= -0.001
 tau_gap_1_2: tau[1,2] - 5e+06 R[1,2] - tau_star_1 <= -0.001
 tau_gap_1_3: tau[1,3] - 5e+06 R[1,3] - tau_star_1 <= -0.001
 tau_gap_1_4: tau[1,4] - 5e+06 R[1,4] - tau_star_1 <= -0.001
 tau_gap_1_5: tau[1,5] - 5e+06 R[1,5] - tau_star_1 <= -0.001
 tau_gap_1_6: tau[1,6] - 5e+06 R[1,6] - tau_star_1 <= -0.001
 tau_gap_1_7: tau[1,7] - 5e+06 R[1,7] - tau_star_1 <= -0.001
 one_R_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4] + R[1,5] + R[1,6]
   + R[1,7] = 1
 Define_Q_hat_1: - 1313.069377500148 R[1,0] - 1328.628503096434 R[1,1]
   - 1336.626358696616 R[1,2] - 1370.181493682129 R[1,3]
   - 1370.06726532377 R[1,4] - 1384.047740730319 R[1,5]
   - 1383.690383614796 R[1,6] - 1383.490833374612 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 73.21545237950482 R[1,0] - 157.5379217437185 R[1,1]
   - 297.9743840004576 R[1,2] - 445.6812203430208 R[1,3]
   - 599.1086602799331 R[1,4] - 754.87441016278 R[1,5]
   - 910.5808709106293 R[1,6] - 1068.262475273545 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 1311.813316046512 R[1,0] - 1227.490846682299 R[1,1]
   - 1087.054384425559 R[1,2] - 939.3475480829961 R[1,3]
   - 785.9201081460837 R[1,4] - 630.1543582632369 R[1,5]
   - 474.4478975153876 R[1,6] - 316.7662931524724 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 600 Sold_0[1] + 400 Left_0[1] + 600 Lost_0[1]
   - 600 Sold_1[1] + 400 Left_1[1] + 600 Lost_1[1] + profits_vars[1] = 0
 R58: - 100 alphas[0] - alphas[1] - 2.240393122426338 alphas[2] - alphas[3]
   + f_var[2] = 0
 Q0_upper_bound_2: - 2439.950183572415 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_2_0: - 100 betas[0,0] - betas[0,1]
   - 2.240393122426338 betas[0,2] - betas[0,3] + tau[2,0] = 0
 tau_computation_2_1: - 100 betas[1,0] - betas[1,1]
   - 2.240393122426338 betas[1,2] - betas[1,3] + tau[2,1] = 0
 tau_computation_2_2: - 100 betas[2,0] - betas[2,1]
   - 2.240393122426338 betas[2,2] - betas[2,3] + tau[2,2] = 0
 tau_computation_2_3: - 100 betas[3,0] - betas[3,1]
   - 2.240393122426338 betas[3,2] - betas[3,3] + tau[2,3] = 0
 tau_computation_2_4: - 100 betas[4,0] - betas[4,1]
   - 2.240393122426338 betas[4,2] - betas[4,3] + tau[2,4] = 0
 tau_computation_2_5: - 100 betas[5,0] - betas[5,1]
   - 2.240393122426338 betas[5,2] - betas[5,3] + tau[2,5] = 0
 tau_computation_2_6: - 100 betas[6,0] - betas[6,1]
   - 2.240393122426338 betas[6,2] - betas[6,3] + tau[2,6] = 0
 tau_computation_2_7: - 100 betas[7,0] - betas[7,1]
   - 2.240393122426338 betas[7,2] - betas[7,3] + tau[2,7] = 0
 tau_gap_2_0: tau[2,0] - 5e+06 R[2,0] - tau_star_2 <= -0.001
 tau_gap_2_1: tau[2,1] - 5e+06 R[2,1] - tau_star_2 <= -0.001
 tau_gap_2_2: tau[2,2] - 5e+06 R[2,2] - tau_star_2 <= -0.001
 tau_gap_2_3: tau[2,3] - 5e+06 R[2,3] - tau_star_2 <= -0.001
 tau_gap_2_4: tau[2,4] - 5e+06 R[2,4] - tau_star_2 <= -0.001
 tau_gap_2_5: tau[2,5] - 5e+06 R[2,5] - tau_star_2 <= -0.001
 tau_gap_2_6: tau[2,6] - 5e+06 R[2,6] - tau_star_2 <= -0.001
 tau_gap_2_7: tau[2,7] - 5e+06 R[2,7] - tau_star_2 <= -0.001
 one_R_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4] + R[2,5] + R[2,6]
   + R[2,7] = 1
 Define_Q_hat_2: - 1347.728424687034 R[2,0] - 1365.324931387062 R[2,1]
   - 1365.503152646221 R[2,2] - 1396.960415751585 R[2,3]
   - 1397.087606197259 R[2,4] - 1401.439729913676 R[2,5]
   - 1402.952999388816 R[2,6] - 1401.211673186868 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 81.61435377087084 R[2,0] - 174.9029840954004 R[2,1]
   - 317.6387924180795 R[2,2] - 467.4128803056907 R[2,3]
   - 622.1876988929929 R[2,4] - 778.1656307812598 R[2,5]
   - 934.5101369353068 R[2,6] - 1087.275011080995 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1318.854363175526 R[2,0] - 1225.565732850997 R[2,1]
   - 1082.829924528318 R[2,2] - 933.0558366407067 R[2,3]
   - 778.2810180534046 R[2,4] - 622.3030861651378 R[2,5]
   - 465.9585800110906 R[2,6] - 313.1937058654025 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 600 Sold_0[2] + 400 Left_0[2] + 600 Lost_0[2]
   - 600 Sold_1[2] + 400 Left_1[2] + 600 Lost_1[2] + profits_vars[2] = 0
 R87: - 100 alphas[0] - alphas[1] - 2.59228039504429 alphas[2] - alphas[3]
   + f_var[3] = 0
 Q0_upper_bound_3: - 2439.950183572415 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_3_0: - 100 betas[0,0] - betas[0,1]
   - 2.59228039504429 betas[0,2] - betas[0,3] + tau[3,0] = 0
 tau_computation_3_1: - 100 betas[1,0] - betas[1,1]
   - 2.59228039504429 betas[1,2] - betas[1,3] + tau[3,1] = 0
 tau_computation_3_2: - 100 betas[2,0] - betas[2,1]
   - 2.59228039504429 betas[2,2] - betas[2,3] + tau[3,2] = 0
 tau_computation_3_3: - 100 betas[3,0] - betas[3,1]
   - 2.59228039504429 betas[3,2] - betas[3,3] + tau[3,3] = 0
 tau_computation_3_4: - 100 betas[4,0] - betas[4,1]
   - 2.59228039504429 betas[4,2] - betas[4,3] + tau[3,4] = 0
 tau_computation_3_5: - 100 betas[5,0] - betas[5,1]
   - 2.59228039504429 betas[5,2] - betas[5,3] + tau[3,5] = 0
 tau_computation_3_6: - 100 betas[6,0] - betas[6,1]
   - 2.59228039504429 betas[6,2] - betas[6,3] + tau[3,6] = 0
 tau_computation_3_7: - 100 betas[7,0] - betas[7,1]
   - 2.59228039504429 betas[7,2] - betas[7,3] + tau[3,7] = 0
 tau_gap_3_0: tau[3,0] - 5e+06 R[3,0] - tau_star_3 <= -0.001
 tau_gap_3_1: tau[3,1] - 5e+06 R[3,1] - tau_star_3 <= -0.001
 tau_gap_3_2: tau[3,2] - 5e+06 R[3,2] - tau_star_3 <= -0.001
 tau_gap_3_3: tau[3,3] - 5e+06 R[3,3] - tau_star_3 <= -0.001
 tau_gap_3_4: tau[3,4] - 5e+06 R[3,4] - tau_star_3 <= -0.001
 tau_gap_3_5: tau[3,5] - 5e+06 R[3,5] - tau_star_3 <= -0.001
 tau_gap_3_6: tau[3,6] - 5e+06 R[3,6] - tau_star_3 <= -0.001
 tau_gap_3_7: tau[3,7] - 5e+06 R[3,7] - tau_star_3 <= -0.001
 one_R_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4] + R[3,5] + R[3,6]
   + R[3,7] = 1
 Define_Q_hat_3: - 1233.762715617873 R[3,0] - 1261.705211601812 R[3,1]
   - 1271.91959007735 R[3,2] - 1264.597859544679 R[3,3]
   - 1264.10583397753 R[3,4] - 1262.31974999384 R[3,5]
   - 1265.93940421012 R[3,6] - 1266.018575733685 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 53.99712738528318 R[3,0] - 119.3758196209804 R[3,1]
   - 241.9775953857487 R[3,2] - 379.3474574693201 R[3,3]
   - 523.3187989587616 R[3,4] - 668.9156063886583 R[3,5]
   - 817.6339110010226 R[3,6] - 966.400270026445 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 1210.842186081139 R[3,0] - 1145.463493845442 R[3,1]
   - 1022.861718080673 R[3,2] - 885.491855997102 R[3,3]
   - 741.5205145076607 R[3,4] - 595.9237070777638 R[3,5]
   - 447.2054024653995 R[3,6] - 298.4390434399771 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 600 Sold_0[3] + 400 Left_0[3] + 600 Lost_0[3]
   - 600 Sold_1[3] + 400 Left_1[3] + 600 Lost_1[3] + profits_vars[3] = 0
 R116: - 100 alphas[0] - alphas[1] - 2.237455437908488 alphas[2]
   - alphas[3] + f_var[4] = 0
 Q0_upper_bound_4: - 2439.950183572415 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_4_0: - 100 betas[0,0] - betas[0,1]
   - 2.237455437908488 betas[0,2] - betas[0,3] + tau[4,0] = 0
 tau_computation_4_1: - 100 betas[1,0] - betas[1,1]
   - 2.237455437908488 betas[1,2] - betas[1,3] + tau[4,1] = 0
 tau_computation_4_2: - 100 betas[2,0] - betas[2,1]
   - 2.237455437908488 betas[2,2] - betas[2,3] + tau[4,2] = 0
 tau_computation_4_3: - 100 betas[3,0] - betas[3,1]
   - 2.237455437908488 betas[3,2] - betas[3,3] + tau[4,3] = 0
 tau_computation_4_4: - 100 betas[4,0] - betas[4,1]
   - 2.237455437908488 betas[4,2] - betas[4,3] + tau[4,4] = 0
 tau_computation_4_5: - 100 betas[5,0] - betas[5,1]
   - 2.237455437908488 betas[5,2] - betas[5,3] + tau[4,5] = 0
 tau_computation_4_6: - 100 betas[6,0] - betas[6,1]
   - 2.237455437908488 betas[6,2] - betas[6,3] + tau[4,6] = 0
 tau_computation_4_7: - 100 betas[7,0] - betas[7,1]
   - 2.237455437908488 betas[7,2] - betas[7,3] + tau[4,7] = 0
 tau_gap_4_0: tau[4,0] - 5e+06 R[4,0] - tau_star_4 <= -0.001
 tau_gap_4_1: tau[4,1] - 5e+06 R[4,1] - tau_star_4 <= -0.001
 tau_gap_4_2: tau[4,2] - 5e+06 R[4,2] - tau_star_4 <= -0.001
 tau_gap_4_3: tau[4,3] - 5e+06 R[4,3] - tau_star_4 <= -0.001
 tau_gap_4_4: tau[4,4] - 5e+06 R[4,4] - tau_star_4 <= -0.001
 tau_gap_4_5: tau[4,5] - 5e+06 R[4,5] - tau_star_4 <= -0.001
 tau_gap_4_6: tau[4,6] - 5e+06 R[4,6] - tau_star_4 <= -0.001
 tau_gap_4_7: tau[4,7] - 5e+06 R[4,7] - tau_star_4 <= -0.001
 one_R_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4] + R[4,5] + R[4,6]
   + R[4,7] = 1
 Define_Q_hat_4: - 1339.51303081477 R[4,0] - 1329.221296579172 R[4,1]
   - 1331.942771466494 R[4,2] - 1298.853685280946 R[4,3]
   - 1298.985542695456 R[4,4] - 1289.18443392852 R[4,5]
   - 1289.069098168646 R[4,6] - 1287.215295496354 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 79.62352342137159 R[4,0] - 168.2585305379253 R[4,1]
   - 297.9024988186243 R[4,2] - 434.0114956251201 R[4,3]
   - 575.3385108270288 R[4,4] - 717.6362335267687 R[4,5]
   - 860.8298790445028 R[4,6] - 1001.088464092355 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 1205.629126794681 R[4,0] - 1116.994119678127 R[4,1]
   - 987.3501513974285 R[4,2] - 851.2411545909326 R[4,3]
   - 709.9141393890238 R[4,4] - 567.6164166892839 R[4,5]
   - 424.4227711715498 R[4,6] - 284.1641861236983 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 600 Sold_0[4] + 400 Left_0[4] + 600 Lost_0[4]
   - 600 Sold_1[4] + 400 Left_1[4] + 600 Lost_1[4] + profits_vars[4] = 0
 R145: - 100 alphas[0] - alphas[1] - 2.635966358787952 alphas[2]
   - alphas[3] + f_var[5] = 0
 Q0_upper_bound_5: - 2439.950183572415 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_5_0: - 100 betas[0,0] - betas[0,1]
   - 2.635966358787952 betas[0,2] - betas[0,3] + tau[5,0] = 0
 tau_computation_5_1: - 100 betas[1,0] - betas[1,1]
   - 2.635966358787952 betas[1,2] - betas[1,3] + tau[5,1] = 0
 tau_computation_5_2: - 100 betas[2,0] - betas[2,1]
   - 2.635966358787952 betas[2,2] - betas[2,3] + tau[5,2] = 0
 tau_computation_5_3: - 100 betas[3,0] - betas[3,1]
   - 2.635966358787952 betas[3,2] - betas[3,3] + tau[5,3] = 0
 tau_computation_5_4: - 100 betas[4,0] - betas[4,1]
   - 2.635966358787952 betas[4,2] - betas[4,3] + tau[5,4] = 0
 tau_computation_5_5: - 100 betas[5,0] - betas[5,1]
   - 2.635966358787952 betas[5,2] - betas[5,3] + tau[5,5] = 0
 tau_computation_5_6: - 100 betas[6,0] - betas[6,1]
   - 2.635966358787952 betas[6,2] - betas[6,3] + tau[5,6] = 0
 tau_computation_5_7: - 100 betas[7,0] - betas[7,1]
   - 2.635966358787952 betas[7,2] - betas[7,3] + tau[5,7] = 0
 tau_gap_5_0: tau[5,0] - 5e+06 R[5,0] - tau_star_5 <= -0.001
 tau_gap_5_1: tau[5,1] - 5e+06 R[5,1] - tau_star_5 <= -0.001
 tau_gap_5_2: tau[5,2] - 5e+06 R[5,2] - tau_star_5 <= -0.001
 tau_gap_5_3: tau[5,3] - 5e+06 R[5,3] - tau_star_5 <= -0.001
 tau_gap_5_4: tau[5,4] - 5e+06 R[5,4] - tau_star_5 <= -0.001
 tau_gap_5_5: tau[5,5] - 5e+06 R[5,5] - tau_star_5 <= -0.001
 tau_gap_5_6: tau[5,6] - 5e+06 R[5,6] - tau_star_5 <= -0.001
 tau_gap_5_7: tau[5,7] - 5e+06 R[5,7] - tau_star_5 <= -0.001
 one_R_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4] + R[5,5] + R[5,6]
   + R[5,7] = 1
 Define_Q_hat_5: - 1285.661197098168 R[5,0] - 1258.872997928785 R[5,1]
   - 1271.491573176483 R[5,2] - 1241.816506545344 R[5,3]
   - 1241.609592631933 R[5,4] - 1235.337914890154 R[5,5]
   - 1237.256857935524 R[5,6] - 1235.546401008579 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 66.57364819857214 R[5,0] - 140.0475290087377 R[5,1]
   - 256.5566557998753 R[5,2] - 387.464139985612 R[5,3]
   - 525.6343393988873 R[5,4] - 665.0491688482132 R[5,5]
   - 808.138828893733 R[5,6] - 949.3483661035787 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 1168.747367415433 R[5,0] - 1095.273486605267 R[5,1]
   - 978.76435981413 R[5,2] - 847.8568756283934 R[5,3]
   - 709.6866762151179 R[5,4] - 570.271846765792 R[5,5]
   - 427.1821867202722 R[5,6] - 285.9726495104265 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 600 Sold_0[5] + 400 Left_0[5] + 600 Lost_0[5]
   - 600 Sold_1[5] + 400 Left_1[5] + 600 Lost_1[5] + profits_vars[5] = 0
 R174: - 100 alphas[0] - alphas[1] - 2.82852598902934 alphas[2] - alphas[3]
   + f_var[6] = 0
 Q0_upper_bound_6: - 2439.950183572415 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_6_0: - 100 betas[0,0] - betas[0,1]
   - 2.82852598902934 betas[0,2] - betas[0,3] + tau[6,0] = 0
 tau_computation_6_1: - 100 betas[1,0] - betas[1,1]
   - 2.82852598902934 betas[1,2] - betas[1,3] + tau[6,1] = 0
 tau_computation_6_2: - 100 betas[2,0] - betas[2,1]
   - 2.82852598902934 betas[2,2] - betas[2,3] + tau[6,2] = 0
 tau_computation_6_3: - 100 betas[3,0] - betas[3,1]
   - 2.82852598902934 betas[3,2] - betas[3,3] + tau[6,3] = 0
 tau_computation_6_4: - 100 betas[4,0] - betas[4,1]
   - 2.82852598902934 betas[4,2] - betas[4,3] + tau[6,4] = 0
 tau_computation_6_5: - 100 betas[5,0] - betas[5,1]
   - 2.82852598902934 betas[5,2] - betas[5,3] + tau[6,5] = 0
 tau_computation_6_6: - 100 betas[6,0] - betas[6,1]
   - 2.82852598902934 betas[6,2] - betas[6,3] + tau[6,6] = 0
 tau_computation_6_7: - 100 betas[7,0] - betas[7,1]
   - 2.82852598902934 betas[7,2] - betas[7,3] + tau[6,7] = 0
 tau_gap_6_0: tau[6,0] - 5e+06 R[6,0] - tau_star_6 <= -0.001
 tau_gap_6_1: tau[6,1] - 5e+06 R[6,1] - tau_star_6 <= -0.001
 tau_gap_6_2: tau[6,2] - 5e+06 R[6,2] - tau_star_6 <= -0.001
 tau_gap_6_3: tau[6,3] - 5e+06 R[6,3] - tau_star_6 <= -0.001
 tau_gap_6_4: tau[6,4] - 5e+06 R[6,4] - tau_star_6 <= -0.001
 tau_gap_6_5: tau[6,5] - 5e+06 R[6,5] - tau_star_6 <= -0.001
 tau_gap_6_6: tau[6,6] - 5e+06 R[6,6] - tau_star_6 <= -0.001
 tau_gap_6_7: tau[6,7] - 5e+06 R[6,7] - tau_star_6 <= -0.001
 one_R_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4] + R[6,5] + R[6,6]
   + R[6,7] = 1
 Define_Q_hat_6: - 1264.901907047903 R[6,0] - 1256.641397302901 R[6,1]
   - 1269.561936991104 R[6,2] - 1217.293767971039 R[6,3]
   - 1217.707348715403 R[6,4] - 1213.195543015619 R[6,5]
   - 1211.540606404244 R[6,6] - 1212.409299514012 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 61.54306472214842 R[6,0] - 131.4684891067634 R[6,1]
   - 250.4499298179138 R[6,2] - 378.9909330066727 R[6,3]
   - 509.8694925152957 R[6,4] - 648.2272563359381 R[6,5]
   - 787.4158082448697 R[6,6] - 930.6163277224031 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 1151.726305583351 R[6,0] - 1081.800881198736 R[6,1]
   - 962.8194404875858 R[6,2] - 834.2784372988269 R[6,3]
   - 703.3998777902038 R[6,4] - 565.0421139695613 R[6,5]
   - 425.8535620606299 R[6,6] - 282.6530425830965 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 600 Sold_0[6] + 400 Left_0[6] + 600 Lost_0[6]
   - 600 Sold_1[6] + 400 Left_1[6] + 600 Lost_1[6] + profits_vars[6] = 0
 R203: - 100 alphas[0] - alphas[1] - 2.128294992697569 alphas[2]
   - alphas[3] + f_var[7] = 0
 Q0_upper_bound_7: - 2439.950183572415 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_7_0: - 100 betas[0,0] - betas[0,1]
   - 2.128294992697569 betas[0,2] - betas[0,3] + tau[7,0] = 0
 tau_computation_7_1: - 100 betas[1,0] - betas[1,1]
   - 2.128294992697569 betas[1,2] - betas[1,3] + tau[7,1] = 0
 tau_computation_7_2: - 100 betas[2,0] - betas[2,1]
   - 2.128294992697569 betas[2,2] - betas[2,3] + tau[7,2] = 0
 tau_computation_7_3: - 100 betas[3,0] - betas[3,1]
   - 2.128294992697569 betas[3,2] - betas[3,3] + tau[7,3] = 0
 tau_computation_7_4: - 100 betas[4,0] - betas[4,1]
   - 2.128294992697569 betas[4,2] - betas[4,3] + tau[7,4] = 0
 tau_computation_7_5: - 100 betas[5,0] - betas[5,1]
   - 2.128294992697569 betas[5,2] - betas[5,3] + tau[7,5] = 0
 tau_computation_7_6: - 100 betas[6,0] - betas[6,1]
   - 2.128294992697569 betas[6,2] - betas[6,3] + tau[7,6] = 0
 tau_computation_7_7: - 100 betas[7,0] - betas[7,1]
   - 2.128294992697569 betas[7,2] - betas[7,3] + tau[7,7] = 0
 tau_gap_7_0: tau[7,0] - 5e+06 R[7,0] - tau_star_7 <= -0.001
 tau_gap_7_1: tau[7,1] - 5e+06 R[7,1] - tau_star_7 <= -0.001
 tau_gap_7_2: tau[7,2] - 5e+06 R[7,2] - tau_star_7 <= -0.001
 tau_gap_7_3: tau[7,3] - 5e+06 R[7,3] - tau_star_7 <= -0.001
 tau_gap_7_4: tau[7,4] - 5e+06 R[7,4] - tau_star_7 <= -0.001
 tau_gap_7_5: tau[7,5] - 5e+06 R[7,5] - tau_star_7 <= -0.001
 tau_gap_7_6: tau[7,6] - 5e+06 R[7,6] - tau_star_7 <= -0.001
 tau_gap_7_7: tau[7,7] - 5e+06 R[7,7] - tau_star_7 <= -0.001
 one_R_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4] + R[7,5] + R[7,6]
   + R[7,7] = 1
 Define_Q_hat_7: - 1332.543516964268 R[7,0] - 1348.68839850661 R[7,1]
   - 1342.174180725441 R[7,2] - 1304.886877301685 R[7,3]
   - 1305.418222533907 R[7,4] - 1287.275504758111 R[7,5]
   - 1288.719360961795 R[7,6] - 1290.137031262155 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 77.93460626042103 R[7,0] - 167.243387974712 R[7,1]
   - 295.174486712393 R[7,2] - 432.9857522853443 R[7,3]
   - 573.3628558107237 R[7,4] - 716.8214428038407 R[7,5]
   - 861.001490256695 R[7,6] - 1003.105260369112 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 1212.772921294161 R[7,0] - 1123.46413957987 R[7,1]
   - 995.5330408421894 R[7,2] - 857.7217752692379 R[7,3]
   - 717.3446717438585 R[7,4] - 573.8860847507415 R[7,5]
   - 429.7060372978872 R[7,6] - 287.6022671854701 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 600 Sold_0[7] + 400 Left_0[7] + 600 Lost_0[7]
   - 600 Sold_1[7] + 400 Left_1[7] + 600 Lost_1[7] + profits_vars[7] = 0
 R232: - 100 alphas[0] - alphas[1] - 3.38494423874004 alphas[2] - alphas[3]
   + f_var[8] = 0
 Q0_upper_bound_8: - 2439.950183572415 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_8_0: - 100 betas[0,0] - betas[0,1]
   - 3.38494423874004 betas[0,2] - betas[0,3] + tau[8,0] = 0
 tau_computation_8_1: - 100 betas[1,0] - betas[1,1]
   - 3.38494423874004 betas[1,2] - betas[1,3] + tau[8,1] = 0
 tau_computation_8_2: - 100 betas[2,0] - betas[2,1]
   - 3.38494423874004 betas[2,2] - betas[2,3] + tau[8,2] = 0
 tau_computation_8_3: - 100 betas[3,0] - betas[3,1]
   - 3.38494423874004 betas[3,2] - betas[3,3] + tau[8,3] = 0
 tau_computation_8_4: - 100 betas[4,0] - betas[4,1]
   - 3.38494423874004 betas[4,2] - betas[4,3] + tau[8,4] = 0
 tau_computation_8_5: - 100 betas[5,0] - betas[5,1]
   - 3.38494423874004 betas[5,2] - betas[5,3] + tau[8,5] = 0
 tau_computation_8_6: - 100 betas[6,0] - betas[6,1]
   - 3.38494423874004 betas[6,2] - betas[6,3] + tau[8,6] = 0
 tau_computation_8_7: - 100 betas[7,0] - betas[7,1]
   - 3.38494423874004 betas[7,2] - betas[7,3] + tau[8,7] = 0
 tau_gap_8_0: tau[8,0] - 5e+06 R[8,0] - tau_star_8 <= -0.001
 tau_gap_8_1: tau[8,1] - 5e+06 R[8,1] - tau_star_8 <= -0.001
 tau_gap_8_2: tau[8,2] - 5e+06 R[8,2] - tau_star_8 <= -0.001
 tau_gap_8_3: tau[8,3] - 5e+06 R[8,3] - tau_star_8 <= -0.001
 tau_gap_8_4: tau[8,4] - 5e+06 R[8,4] - tau_star_8 <= -0.001
 tau_gap_8_5: tau[8,5] - 5e+06 R[8,5] - tau_star_8 <= -0.001
 tau_gap_8_6: tau[8,6] - 5e+06 R[8,6] - tau_star_8 <= -0.001
 tau_gap_8_7: tau[8,7] - 5e+06 R[8,7] - tau_star_8 <= -0.001
 one_R_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4] + R[8,5] + R[8,6]
   + R[8,7] = 1
 Define_Q_hat_8: - 1264.954341140065 R[8,0] - 1208.377747629048 R[8,1]
   - 1207.214079144577 R[8,2] - 1293.9403884295 R[8,3]
   - 1294.746933557004 R[8,4] - 1286.732322785419 R[8,5]
   - 1286.089654782524 R[8,6] - 1286.028859316417 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 61.55577103710379 R[8,0] - 127.0369769954401 R[8,1]
   - 203.468163403897 R[8,2] - 344.4844505380227 R[8,3]
   - 493.0968596894205 R[8,4] - 648.24970006427 R[8,5]
   - 808.3489285561668 R[8,6] - 966.7427315486208 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 1224.202401217163 R[8,0] - 1158.721195258827 R[8,1]
   - 1082.29000885037 R[8,2] - 941.2737217162447 R[8,3]
   - 792.6613125648469 R[8,4] - 637.5084721899974 R[8,5]
   - 477.4092436981006 R[8,6] - 319.0154407056466 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 600 Sold_0[8] + 400 Left_0[8] + 600 Lost_0[8]
   - 600 Sold_1[8] + 400 Left_1[8] + 600 Lost_1[8] + profits_vars[8] = 0
 R261: - 100 alphas[0] - alphas[1] - 3.13320290841315 alphas[2] - alphas[3]
   + f_var[9] = 0
 Q0_upper_bound_9: - 2439.950183572415 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_9_0: - 100 betas[0,0] - betas[0,1]
   - 3.13320290841315 betas[0,2] - betas[0,3] + tau[9,0] = 0
 tau_computation_9_1: - 100 betas[1,0] - betas[1,1]
   - 3.13320290841315 betas[1,2] - betas[1,3] + tau[9,1] = 0
 tau_computation_9_2: - 100 betas[2,0] - betas[2,1]
   - 3.13320290841315 betas[2,2] - betas[2,3] + tau[9,2] = 0
 tau_computation_9_3: - 100 betas[3,0] - betas[3,1]
   - 3.13320290841315 betas[3,2] - betas[3,3] + tau[9,3] = 0
 tau_computation_9_4: - 100 betas[4,0] - betas[4,1]
   - 3.13320290841315 betas[4,2] - betas[4,3] + tau[9,4] = 0
 tau_computation_9_5: - 100 betas[5,0] - betas[5,1]
   - 3.13320290841315 betas[5,2] - betas[5,3] + tau[9,5] = 0
 tau_computation_9_6: - 100 betas[6,0] - betas[6,1]
   - 3.13320290841315 betas[6,2] - betas[6,3] + tau[9,6] = 0
 tau_computation_9_7: - 100 betas[7,0] - betas[7,1]
   - 3.13320290841315 betas[7,2] - betas[7,3] + tau[9,7] = 0
 tau_gap_9_0: tau[9,0] - 5e+06 R[9,0] - tau_star_9 <= -0.001
 tau_gap_9_1: tau[9,1] - 5e+06 R[9,1] - tau_star_9 <= -0.001
 tau_gap_9_2: tau[9,2] - 5e+06 R[9,2] - tau_star_9 <= -0.001
 tau_gap_9_3: tau[9,3] - 5e+06 R[9,3] - tau_star_9 <= -0.001
 tau_gap_9_4: tau[9,4] - 5e+06 R[9,4] - tau_star_9 <= -0.001
 tau_gap_9_5: tau[9,5] - 5e+06 R[9,5] - tau_star_9 <= -0.001
 tau_gap_9_6: tau[9,6] - 5e+06 R[9,6] - tau_star_9 <= -0.001
 tau_gap_9_7: tau[9,7] - 5e+06 R[9,7] - tau_star_9 <= -0.001
 one_R_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4] + R[9,5] + R[9,6]
   + R[9,7] = 1
 Define_Q_hat_9: - 1307.556511440617 R[9,0] - 1246.048537596501 R[9,1]
   - 1245.066473202745 R[9,2] - 1209.078576842173 R[9,3]
   - 1208.540644921419 R[9,4] - 1209.04911004188 R[9,5]
   - 1203.908824468143 R[9,6] - 1202.947455147311 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 71.87952360546365 R[9,0] - 147.6957512828164 R[9,1]
   - 236.3870700429292 R[9,2] - 362.2277887564559 R[9,3]
   - 500.8821273877707 R[9,4] - 638.8279307420383 R[9,5]
   - 778.79599133199 R[9,6] - 922.8103899857904 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 1132.055207568583 R[9,0] - 1056.23897989123 R[9,1]
   - 967.5476611311177 R[9,2] - 841.706942417591 R[9,3]
   - 703.0526037862763 R[9,4] - 565.1068004320086 R[9,5]
   - 425.1387398420569 R[9,6] - 281.1243411882566 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 600 Sold_0[9] + 400 Left_0[9] + 600 Lost_0[9]
   - 600 Sold_1[9] + 400 Left_1[9] + 600 Lost_1[9] + profits_vars[9] = 0
 R290: - 100 alphas[0] - alphas[1] - 2.530778981878891 alphas[2]
   - alphas[3] + f_var[10] = 0
 Q0_upper_bound_10:
   - 2439.950183572415 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 tau_computation_10_0: - 100 betas[0,0] - betas[0,1]
   - 2.530778981878891 betas[0,2] - betas[0,3] + tau[10,0] = 0
 tau_computation_10_1: - 100 betas[1,0] - betas[1,1]
   - 2.530778981878891 betas[1,2] - betas[1,3] + tau[10,1] = 0
 tau_computation_10_2: - 100 betas[2,0] - betas[2,1]
   - 2.530778981878891 betas[2,2] - betas[2,3] + tau[10,2] = 0
 tau_computation_10_3: - 100 betas[3,0] - betas[3,1]
   - 2.530778981878891 betas[3,2] - betas[3,3] + tau[10,3] = 0
 tau_computation_10_4: - 100 betas[4,0] - betas[4,1]
   - 2.530778981878891 betas[4,2] - betas[4,3] + tau[10,4] = 0
 tau_computation_10_5: - 100 betas[5,0] - betas[5,1]
   - 2.530778981878891 betas[5,2] - betas[5,3] + tau[10,5] = 0
 tau_computation_10_6: - 100 betas[6,0] - betas[6,1]
   - 2.530778981878891 betas[6,2] - betas[6,3] + tau[10,6] = 0
 tau_computation_10_7: - 100 betas[7,0] - betas[7,1]
   - 2.530778981878891 betas[7,2] - betas[7,3] + tau[10,7] = 0
 tau_gap_10_0: tau[10,0] - 5e+06 R[10,0] - tau_star_10 <= -0.001
 tau_gap_10_1: tau[10,1] - 5e+06 R[10,1] - tau_star_10 <= -0.001
 tau_gap_10_2: tau[10,2] - 5e+06 R[10,2] - tau_star_10 <= -0.001
 tau_gap_10_3: tau[10,3] - 5e+06 R[10,3] - tau_star_10 <= -0.001
 tau_gap_10_4: tau[10,4] - 5e+06 R[10,4] - tau_star_10 <= -0.001
 tau_gap_10_5: tau[10,5] - 5e+06 R[10,5] - tau_star_10 <= -0.001
 tau_gap_10_6: tau[10,6] - 5e+06 R[10,6] - tau_star_10 <= -0.001
 tau_gap_10_7: tau[10,7] - 5e+06 R[10,7] - tau_star_10 <= -0.001
 one_R_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4] + R[10,5]
   + R[10,6] + R[10,7] = 1
 Define_Q_hat_10: - 1269.668690874512 R[10,0] - 1342.468268353284 R[10,1]
   - 1339.94195074094 R[10,2] - 1317.323924469457 R[10,3]
   - 1317.887363954424 R[10,4] - 1319.622784814633 R[10,5]
   - 1313.700084736271 R[10,6] - 1312.597466627159 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 62.69819593789622 R[10,0] - 141.3092807886965 R[10,1]
   - 281.1005347002001 R[10,2] - 424.1770505792728 R[10,3]
   - 566.3407245536445 R[10,4] - 717.2602521384854 R[10,5]
   - 863.157179828572 R[10,6] - 1012.65095864522 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 1248.984372169221 R[10,0] - 1170.37328731842 R[10,1]
   - 1030.582033406917 R[10,2] - 887.5055175278442 R[10,3]
   - 745.3418435534725 R[10,4] - 594.4223159686314 R[10,5]
   - 448.5253882785449 R[10,6] - 299.0316094618975 R[10,7] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: - Total_Demand_Up_to_K_minus_1[10] + Q0_var[10]
   + Lost_0_aux_10 = 0
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: - Total_Demand_from_k_to_T[10]
   + Q1_plus_left[10] + Lost_0_aux_10 = 0
 Profit_Constraint_10: - 600 Sold_0[10] + 400 Left_0[10] + 600 Lost_0[10]
   - 600 Sold_1[10] + 400 Left_1[10] + 600 Lost_1[10] + profits_vars[10]
   = 0
 R319: - 100 alphas[0] - alphas[1] - 3.046496106933399 alphas[2]
   - alphas[3] + f_var[11] = 0
 Q0_upper_bound_11:
   - 2439.950183572415 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 tau_computation_11_0: - 100 betas[0,0] - betas[0,1]
   - 3.046496106933399 betas[0,2] - betas[0,3] + tau[11,0] = 0
 tau_computation_11_1: - 100 betas[1,0] - betas[1,1]
   - 3.046496106933399 betas[1,2] - betas[1,3] + tau[11,1] = 0
 tau_computation_11_2: - 100 betas[2,0] - betas[2,1]
   - 3.046496106933399 betas[2,2] - betas[2,3] + tau[11,2] = 0
 tau_computation_11_3: - 100 betas[3,0] - betas[3,1]
   - 3.046496106933399 betas[3,2] - betas[3,3] + tau[11,3] = 0
 tau_computation_11_4: - 100 betas[4,0] - betas[4,1]
   - 3.046496106933399 betas[4,2] - betas[4,3] + tau[11,4] = 0
 tau_computation_11_5: - 100 betas[5,0] - betas[5,1]
   - 3.046496106933399 betas[5,2] - betas[5,3] + tau[11,5] = 0
 tau_computation_11_6: - 100 betas[6,0] - betas[6,1]
   - 3.046496106933399 betas[6,2] - betas[6,3] + tau[11,6] = 0
 tau_computation_11_7: - 100 betas[7,0] - betas[7,1]
   - 3.046496106933399 betas[7,2] - betas[7,3] + tau[11,7] = 0
 tau_gap_11_0: tau[11,0] - 5e+06 R[11,0] - tau_star_11 <= -0.001
 tau_gap_11_1: tau[11,1] - 5e+06 R[11,1] - tau_star_11 <= -0.001
 tau_gap_11_2: tau[11,2] - 5e+06 R[11,2] - tau_star_11 <= -0.001
 tau_gap_11_3: tau[11,3] - 5e+06 R[11,3] - tau_star_11 <= -0.001
 tau_gap_11_4: tau[11,4] - 5e+06 R[11,4] - tau_star_11 <= -0.001
 tau_gap_11_5: tau[11,5] - 5e+06 R[11,5] - tau_star_11 <= -0.001
 tau_gap_11_6: tau[11,6] - 5e+06 R[11,6] - tau_star_11 <= -0.001
 tau_gap_11_7: tau[11,7] - 5e+06 R[11,7] - tau_star_11 <= -0.001
 one_R_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4] + R[11,5]
   + R[11,6] + R[11,7] = 1
 Define_Q_hat_11: - 1278.265908159761 R[11,0] - 1258.656413561333 R[11,1]
   - 1246.036645584056 R[11,2] - 1227.447448614683 R[11,3]
   - 1227.130497839967 R[11,4] - 1217.697958739626 R[11,5]
   - 1218.365548809979 R[11,6] - 1217.354640481758 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 64.78155327399368 R[11,0] - 137.0446936196557 R[11,1]
   - 220.8456302279648 R[11,2] - 351.6944513154151 R[11,3]
   - 493.638677475697 R[11,4] - 635.9179756971889 R[11,5]
   - 782.2199128354056 R[11,6] - 926.494872266042 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 1152.902559624203 R[11,0] - 1080.639419278541 R[11,1]
   - 996.8384826702314 R[11,2] - 865.9896615827812 R[11,3]
   - 724.0454354224993 R[11,4] - 581.7661372010075 R[11,5]
   - 435.4642000627908 R[11,6] - 291.1892406321544 R[11,7] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: - Total_Demand_Up_to_K_minus_1[11] + Q0_var[11]
   + Lost_0_aux_11 = 0
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: - Total_Demand_from_k_to_T[11]
   + Q1_plus_left[11] + Lost_0_aux_11 = 0
 Profit_Constraint_11: - 600 Sold_0[11] + 400 Left_0[11] + 600 Lost_0[11]
   - 600 Sold_1[11] + 400 Left_1[11] + 600 Lost_1[11] + profits_vars[11]
   = 0
 R348: - 100 alphas[0] - alphas[1] - 2.187881021516883 alphas[2]
   - alphas[3] + f_var[12] = 0
 Q0_upper_bound_12:
   - 2439.950183572415 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 tau_computation_12_0: - 100 betas[0,0] - betas[0,1]
   - 2.187881021516883 betas[0,2] - betas[0,3] + tau[12,0] = 0
 tau_computation_12_1: - 100 betas[1,0] - betas[1,1]
   - 2.187881021516883 betas[1,2] - betas[1,3] + tau[12,1] = 0
 tau_computation_12_2: - 100 betas[2,0] - betas[2,1]
   - 2.187881021516883 betas[2,2] - betas[2,3] + tau[12,2] = 0
 tau_computation_12_3: - 100 betas[3,0] - betas[3,1]
   - 2.187881021516883 betas[3,2] - betas[3,3] + tau[12,3] = 0
 tau_computation_12_4: - 100 betas[4,0] - betas[4,1]
   - 2.187881021516883 betas[4,2] - betas[4,3] + tau[12,4] = 0
 tau_computation_12_5: - 100 betas[5,0] - betas[5,1]
   - 2.187881021516883 betas[5,2] - betas[5,3] + tau[12,5] = 0
 tau_computation_12_6: - 100 betas[6,0] - betas[6,1]
   - 2.187881021516883 betas[6,2] - betas[6,3] + tau[12,6] = 0
 tau_computation_12_7: - 100 betas[7,0] - betas[7,1]
   - 2.187881021516883 betas[7,2] - betas[7,3] + tau[12,7] = 0
 tau_gap_12_0: tau[12,0] - 5e+06 R[12,0] - tau_star_12 <= -0.001
 tau_gap_12_1: tau[12,1] - 5e+06 R[12,1] - tau_star_12 <= -0.001
 tau_gap_12_2: tau[12,2] - 5e+06 R[12,2] - tau_star_12 <= -0.001
 tau_gap_12_3: tau[12,3] - 5e+06 R[12,3] - tau_star_12 <= -0.001
 tau_gap_12_4: tau[12,4] - 5e+06 R[12,4] - tau_star_12 <= -0.001
 tau_gap_12_5: tau[12,5] - 5e+06 R[12,5] - tau_star_12 <= -0.001
 tau_gap_12_6: tau[12,6] - 5e+06 R[12,6] - tau_star_12 <= -0.001
 tau_gap_12_7: tau[12,7] - 5e+06 R[12,7] - tau_star_12 <= -0.001
 one_R_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4] + R[12,5]
   + R[12,6] + R[12,7] = 1
 Define_Q_hat_12: - 1322.632994866458 R[12,0] - 1357.432855643482 R[12,1]
   - 1351.567245186316 R[12,2] - 1352.86597885503 R[12,3]
   - 1352.533281397599 R[12,4] - 1351.657382600457 R[12,5]
   - 1353.265808939468 R[12,6] - 1354.127677989939 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 75.53299673585353 R[12,0] - 164.0527645858088 R[12,1]
   - 298.659444796211 R[12,2] - 443.9196527263028 R[12,3]
   - 596.133744597322 R[12,4] - 747.3318942005521 R[12,5]
   - 899.0064378479092 R[12,6] - 1050.292826261878 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 1279.679832087366 R[12,0] - 1191.160064237411 R[12,1]
   - 1056.553384027008 R[12,2] - 911.2931760969167 R[12,3]
   - 759.0790842258974 R[12,4] - 607.8809346226673 R[12,5]
   - 456.2063909753102 R[12,6] - 304.920002561341 R[12,7] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: - Total_Demand_Up_to_K_minus_1[12] + Q0_var[12]
   + Lost_0_aux_12 = 0
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: - Total_Demand_from_k_to_T[12]
   + Q1_plus_left[12] + Lost_0_aux_12 = 0
 Profit_Constraint_12: - 600 Sold_0[12] + 400 Left_0[12] + 600 Lost_0[12]
   - 600 Sold_1[12] + 400 Left_1[12] + 600 Lost_1[12] + profits_vars[12]
   = 0
 R377: - 100 alphas[0] - alphas[1] - 3.151892991112359 alphas[2]
   - alphas[3] + f_var[13] = 0
 Q0_upper_bound_13:
   - 2439.950183572415 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 tau_computation_13_0: - 100 betas[0,0] - betas[0,1]
   - 3.151892991112359 betas[0,2] - betas[0,3] + tau[13,0] = 0
 tau_computation_13_1: - 100 betas[1,0] - betas[1,1]
   - 3.151892991112359 betas[1,2] - betas[1,3] + tau[13,1] = 0
 tau_computation_13_2: - 100 betas[2,0] - betas[2,1]
   - 3.151892991112359 betas[2,2] - betas[2,3] + tau[13,2] = 0
 tau_computation_13_3: - 100 betas[3,0] - betas[3,1]
   - 3.151892991112359 betas[3,2] - betas[3,3] + tau[13,3] = 0
 tau_computation_13_4: - 100 betas[4,0] - betas[4,1]
   - 3.151892991112359 betas[4,2] - betas[4,3] + tau[13,4] = 0
 tau_computation_13_5: - 100 betas[5,0] - betas[5,1]
   - 3.151892991112359 betas[5,2] - betas[5,3] + tau[13,5] = 0
 tau_computation_13_6: - 100 betas[6,0] - betas[6,1]
   - 3.151892991112359 betas[6,2] - betas[6,3] + tau[13,6] = 0
 tau_computation_13_7: - 100 betas[7,0] - betas[7,1]
   - 3.151892991112359 betas[7,2] - betas[7,3] + tau[13,7] = 0
 tau_gap_13_0: tau[13,0] - 5e+06 R[13,0] - tau_star_13 <= -0.001
 tau_gap_13_1: tau[13,1] - 5e+06 R[13,1] - tau_star_13 <= -0.001
 tau_gap_13_2: tau[13,2] - 5e+06 R[13,2] - tau_star_13 <= -0.001
 tau_gap_13_3: tau[13,3] - 5e+06 R[13,3] - tau_star_13 <= -0.001
 tau_gap_13_4: tau[13,4] - 5e+06 R[13,4] - tau_star_13 <= -0.001
 tau_gap_13_5: tau[13,5] - 5e+06 R[13,5] - tau_star_13 <= -0.001
 tau_gap_13_6: tau[13,6] - 5e+06 R[13,6] - tau_star_13 <= -0.001
 tau_gap_13_7: tau[13,7] - 5e+06 R[13,7] - tau_star_13 <= -0.001
 one_R_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4] + R[13,5]
   + R[13,6] + R[13,7] = 1
 Define_Q_hat_13: - 1265.82297584561 R[13,0] - 1271.217634513555 R[13,1]
   - 1249.422410458548 R[13,2] - 1260.124936001666 R[13,3]
   - 1259.545682329363 R[13,4] - 1256.475503566814 R[13,5]
   - 1255.237678023657 R[13,6] - 1255.239501125682 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 61.76626664587376 R[13,0] - 133.18475814879 R[13,1]
   - 212.7939872308261 R[13,2] - 349.6006814092767 R[13,3]
   - 499.2986199050198 R[13,4] - 648.3578573328049 R[13,5]
   - 800.0235506236256 R[13,6] - 952.2683452144238 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 1192.608480814875 R[13,0] - 1121.189989311959 R[13,1]
   - 1041.580760229923 R[13,2] - 904.7740660514723 R[13,3]
   - 755.0761275557293 R[13,4] - 606.0168901279442 R[13,5]
   - 454.3511968371236 R[13,6] - 302.1064022463252 R[13,7] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: - Total_Demand_Up_to_K_minus_1[13] + Q0_var[13]
   + Lost_0_aux_13 = 0
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: - Total_Demand_from_k_to_T[13]
   + Q1_plus_left[13] + Lost_0_aux_13 = 0
 Profit_Constraint_13: - 600 Sold_0[13] + 400 Left_0[13] + 600 Lost_0[13]
   - 600 Sold_1[13] + 400 Left_1[13] + 600 Lost_1[13] + profits_vars[13]
   = 0
 R406: - 100 alphas[0] - alphas[1] - 3.858592395152428 alphas[2]
   - alphas[3] + f_var[14] = 0
 Q0_upper_bound_14:
   - 2439.950183572415 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 tau_computation_14_0: - 100 betas[0,0] - betas[0,1]
   - 3.858592395152428 betas[0,2] - betas[0,3] + tau[14,0] = 0
 tau_computation_14_1: - 100 betas[1,0] - betas[1,1]
   - 3.858592395152428 betas[1,2] - betas[1,3] + tau[14,1] = 0
 tau_computation_14_2: - 100 betas[2,0] - betas[2,1]
   - 3.858592395152428 betas[2,2] - betas[2,3] + tau[14,2] = 0
 tau_computation_14_3: - 100 betas[3,0] - betas[3,1]
   - 3.858592395152428 betas[3,2] - betas[3,3] + tau[14,3] = 0
 tau_computation_14_4: - 100 betas[4,0] - betas[4,1]
   - 3.858592395152428 betas[4,2] - betas[4,3] + tau[14,4] = 0
 tau_computation_14_5: - 100 betas[5,0] - betas[5,1]
   - 3.858592395152428 betas[5,2] - betas[5,3] + tau[14,5] = 0
 tau_computation_14_6: - 100 betas[6,0] - betas[6,1]
   - 3.858592395152428 betas[6,2] - betas[6,3] + tau[14,6] = 0
 tau_computation_14_7: - 100 betas[7,0] - betas[7,1]
   - 3.858592395152428 betas[7,2] - betas[7,3] + tau[14,7] = 0
 tau_gap_14_0: tau[14,0] - 5e+06 R[14,0] - tau_star_14 <= -0.001
 tau_gap_14_1: tau[14,1] - 5e+06 R[14,1] - tau_star_14 <= -0.001
 tau_gap_14_2: tau[14,2] - 5e+06 R[14,2] - tau_star_14 <= -0.001
 tau_gap_14_3: tau[14,3] - 5e+06 R[14,3] - tau_star_14 <= -0.001
 tau_gap_14_4: tau[14,4] - 5e+06 R[14,4] - tau_star_14 <= -0.001
 tau_gap_14_5: tau[14,5] - 5e+06 R[14,5] - tau_star_14 <= -0.001
 tau_gap_14_6: tau[14,6] - 5e+06 R[14,6] - tau_star_14 <= -0.001
 tau_gap_14_7: tau[14,7] - 5e+06 R[14,7] - tau_star_14 <= -0.001
 one_R_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4] + R[14,5]
   + R[14,6] + R[14,7] = 1
 Define_Q_hat_14: - 1294.930180210403 R[14,0] - 1249.176619077158 R[14,1]
   - 1239.368469833272 R[14,2] - 1180.906358084208 R[14,3]
   - 1181.417957328435 R[14,4] - 1201.332475156129 R[14,5]
   - 1203.41105546052 R[14,6] - 1204.306762712083 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 68.81979405301902 R[14,0] - 142.8915692829293 R[14,1]
   - 223.2500544287856 R[14,2] - 346.2536098476997 R[14,3]
   - 476.0859004559387 R[14,4] - 617.0679570734912 R[14,5]
   - 762.6585344678876 R[14,6] - 911.5475941772653 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 1134.080386035188 R[14,0] - 1060.008610805278 R[14,1]
   - 979.6501256594217 R[14,2] - 856.6465702405076 R[14,3]
   - 726.8142796322688 R[14,4] - 585.8322230147162 R[14,5]
   - 440.2416456203198 R[14,6] - 291.3525859109421 R[14,7] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: - Total_Demand_Up_to_K_minus_1[14] + Q0_var[14]
   + Lost_0_aux_14 = 0
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: - Total_Demand_from_k_to_T[14]
   + Q1_plus_left[14] + Lost_0_aux_14 = 0
 Profit_Constraint_14: - 600 Sold_0[14] + 400 Left_0[14] + 600 Lost_0[14]
   - 600 Sold_1[14] + 400 Left_1[14] + 600 Lost_1[14] + profits_vars[14]
   = 0
 tau_star_eq_0_0: R[0,0] = 1 -> tau[0,0] - tau_star_0 = 0
 tau_star_eq_0_1: R[0,1] = 1 -> tau[0,1] - tau_star_0 = 0
 tau_star_eq_0_2: R[0,2] = 1 -> tau[0,2] - tau_star_0 = 0
 tau_star_eq_0_3: R[0,3] = 1 -> tau[0,3] - tau_star_0 = 0
 tau_star_eq_0_4: R[0,4] = 1 -> tau[0,4] - tau_star_0 = 0
 tau_star_eq_0_5: R[0,5] = 1 -> tau[0,5] - tau_star_0 = 0
 tau_star_eq_0_6: R[0,6] = 1 -> tau[0,6] - tau_star_0 = 0
 tau_star_eq_0_7: R[0,7] = 1 -> tau[0,7] - tau_star_0 = 0
 tau_star_eq_1_0: R[1,0] = 1 -> tau[1,0] - tau_star_1 = 0
 tau_star_eq_1_1: R[1,1] = 1 -> tau[1,1] - tau_star_1 = 0
 tau_star_eq_1_2: R[1,2] = 1 -> tau[1,2] - tau_star_1 = 0
 tau_star_eq_1_3: R[1,3] = 1 -> tau[1,3] - tau_star_1 = 0
 tau_star_eq_1_4: R[1,4] = 1 -> tau[1,4] - tau_star_1 = 0
 tau_star_eq_1_5: R[1,5] = 1 -> tau[1,5] - tau_star_1 = 0
 tau_star_eq_1_6: R[1,6] = 1 -> tau[1,6] - tau_star_1 = 0
 tau_star_eq_1_7: R[1,7] = 1 -> tau[1,7] - tau_star_1 = 0
 tau_star_eq_2_0: R[2,0] = 1 -> tau[2,0] - tau_star_2 = 0
 tau_star_eq_2_1: R[2,1] = 1 -> tau[2,1] - tau_star_2 = 0
 tau_star_eq_2_2: R[2,2] = 1 -> tau[2,2] - tau_star_2 = 0
 tau_star_eq_2_3: R[2,3] = 1 -> tau[2,3] - tau_star_2 = 0
 tau_star_eq_2_4: R[2,4] = 1 -> tau[2,4] - tau_star_2 = 0
 tau_star_eq_2_5: R[2,5] = 1 -> tau[2,5] - tau_star_2 = 0
 tau_star_eq_2_6: R[2,6] = 1 -> tau[2,6] - tau_star_2 = 0
 tau_star_eq_2_7: R[2,7] = 1 -> tau[2,7] - tau_star_2 = 0
 tau_star_eq_3_0: R[3,0] = 1 -> tau[3,0] - tau_star_3 = 0
 tau_star_eq_3_1: R[3,1] = 1 -> tau[3,1] - tau_star_3 = 0
 tau_star_eq_3_2: R[3,2] = 1 -> tau[3,2] - tau_star_3 = 0
 tau_star_eq_3_3: R[3,3] = 1 -> tau[3,3] - tau_star_3 = 0
 tau_star_eq_3_4: R[3,4] = 1 -> tau[3,4] - tau_star_3 = 0
 tau_star_eq_3_5: R[3,5] = 1 -> tau[3,5] - tau_star_3 = 0
 tau_star_eq_3_6: R[3,6] = 1 -> tau[3,6] - tau_star_3 = 0
 tau_star_eq_3_7: R[3,7] = 1 -> tau[3,7] - tau_star_3 = 0
 tau_star_eq_4_0: R[4,0] = 1 -> tau[4,0] - tau_star_4 = 0
 tau_star_eq_4_1: R[4,1] = 1 -> tau[4,1] - tau_star_4 = 0
 tau_star_eq_4_2: R[4,2] = 1 -> tau[4,2] - tau_star_4 = 0
 tau_star_eq_4_3: R[4,3] = 1 -> tau[4,3] - tau_star_4 = 0
 tau_star_eq_4_4: R[4,4] = 1 -> tau[4,4] - tau_star_4 = 0
 tau_star_eq_4_5: R[4,5] = 1 -> tau[4,5] - tau_star_4 = 0
 tau_star_eq_4_6: R[4,6] = 1 -> tau[4,6] - tau_star_4 = 0
 tau_star_eq_4_7: R[4,7] = 1 -> tau[4,7] - tau_star_4 = 0
 tau_star_eq_5_0: R[5,0] = 1 -> tau[5,0] - tau_star_5 = 0
 tau_star_eq_5_1: R[5,1] = 1 -> tau[5,1] - tau_star_5 = 0
 tau_star_eq_5_2: R[5,2] = 1 -> tau[5,2] - tau_star_5 = 0
 tau_star_eq_5_3: R[5,3] = 1 -> tau[5,3] - tau_star_5 = 0
 tau_star_eq_5_4: R[5,4] = 1 -> tau[5,4] - tau_star_5 = 0
 tau_star_eq_5_5: R[5,5] = 1 -> tau[5,5] - tau_star_5 = 0
 tau_star_eq_5_6: R[5,6] = 1 -> tau[5,6] - tau_star_5 = 0
 tau_star_eq_5_7: R[5,7] = 1 -> tau[5,7] - tau_star_5 = 0
 tau_star_eq_6_0: R[6,0] = 1 -> tau[6,0] - tau_star_6 = 0
 tau_star_eq_6_1: R[6,1] = 1 -> tau[6,1] - tau_star_6 = 0
 tau_star_eq_6_2: R[6,2] = 1 -> tau[6,2] - tau_star_6 = 0
 tau_star_eq_6_3: R[6,3] = 1 -> tau[6,3] - tau_star_6 = 0
 tau_star_eq_6_4: R[6,4] = 1 -> tau[6,4] - tau_star_6 = 0
 tau_star_eq_6_5: R[6,5] = 1 -> tau[6,5] - tau_star_6 = 0
 tau_star_eq_6_6: R[6,6] = 1 -> tau[6,6] - tau_star_6 = 0
 tau_star_eq_6_7: R[6,7] = 1 -> tau[6,7] - tau_star_6 = 0
 tau_star_eq_7_0: R[7,0] = 1 -> tau[7,0] - tau_star_7 = 0
 tau_star_eq_7_1: R[7,1] = 1 -> tau[7,1] - tau_star_7 = 0
 tau_star_eq_7_2: R[7,2] = 1 -> tau[7,2] - tau_star_7 = 0
 tau_star_eq_7_3: R[7,3] = 1 -> tau[7,3] - tau_star_7 = 0
 tau_star_eq_7_4: R[7,4] = 1 -> tau[7,4] - tau_star_7 = 0
 tau_star_eq_7_5: R[7,5] = 1 -> tau[7,5] - tau_star_7 = 0
 tau_star_eq_7_6: R[7,6] = 1 -> tau[7,6] - tau_star_7 = 0
 tau_star_eq_7_7: R[7,7] = 1 -> tau[7,7] - tau_star_7 = 0
 tau_star_eq_8_0: R[8,0] = 1 -> tau[8,0] - tau_star_8 = 0
 tau_star_eq_8_1: R[8,1] = 1 -> tau[8,1] - tau_star_8 = 0
 tau_star_eq_8_2: R[8,2] = 1 -> tau[8,2] - tau_star_8 = 0
 tau_star_eq_8_3: R[8,3] = 1 -> tau[8,3] - tau_star_8 = 0
 tau_star_eq_8_4: R[8,4] = 1 -> tau[8,4] - tau_star_8 = 0
 tau_star_eq_8_5: R[8,5] = 1 -> tau[8,5] - tau_star_8 = 0
 tau_star_eq_8_6: R[8,6] = 1 -> tau[8,6] - tau_star_8 = 0
 tau_star_eq_8_7: R[8,7] = 1 -> tau[8,7] - tau_star_8 = 0
 tau_star_eq_9_0: R[9,0] = 1 -> tau[9,0] - tau_star_9 = 0
 tau_star_eq_9_1: R[9,1] = 1 -> tau[9,1] - tau_star_9 = 0
 tau_star_eq_9_2: R[9,2] = 1 -> tau[9,2] - tau_star_9 = 0
 tau_star_eq_9_3: R[9,3] = 1 -> tau[9,3] - tau_star_9 = 0
 tau_star_eq_9_4: R[9,4] = 1 -> tau[9,4] - tau_star_9 = 0
 tau_star_eq_9_5: R[9,5] = 1 -> tau[9,5] - tau_star_9 = 0
 tau_star_eq_9_6: R[9,6] = 1 -> tau[9,6] - tau_star_9 = 0
 tau_star_eq_9_7: R[9,7] = 1 -> tau[9,7] - tau_star_9 = 0
 tau_star_eq_10_0: R[10,0] = 1 -> tau[10,0] - tau_star_10 = 0
 tau_star_eq_10_1: R[10,1] = 1 -> tau[10,1] - tau_star_10 = 0
 tau_star_eq_10_2: R[10,2] = 1 -> tau[10,2] - tau_star_10 = 0
 tau_star_eq_10_3: R[10,3] = 1 -> tau[10,3] - tau_star_10 = 0
 tau_star_eq_10_4: R[10,4] = 1 -> tau[10,4] - tau_star_10 = 0
 tau_star_eq_10_5: R[10,5] = 1 -> tau[10,5] - tau_star_10 = 0
 tau_star_eq_10_6: R[10,6] = 1 -> tau[10,6] - tau_star_10 = 0
 tau_star_eq_10_7: R[10,7] = 1 -> tau[10,7] - tau_star_10 = 0
 tau_star_eq_11_0: R[11,0] = 1 -> tau[11,0] - tau_star_11 = 0
 tau_star_eq_11_1: R[11,1] = 1 -> tau[11,1] - tau_star_11 = 0
 tau_star_eq_11_2: R[11,2] = 1 -> tau[11,2] - tau_star_11 = 0
 tau_star_eq_11_3: R[11,3] = 1 -> tau[11,3] - tau_star_11 = 0
 tau_star_eq_11_4: R[11,4] = 1 -> tau[11,4] - tau_star_11 = 0
 tau_star_eq_11_5: R[11,5] = 1 -> tau[11,5] - tau_star_11 = 0
 tau_star_eq_11_6: R[11,6] = 1 -> tau[11,6] - tau_star_11 = 0
 tau_star_eq_11_7: R[11,7] = 1 -> tau[11,7] - tau_star_11 = 0
 tau_star_eq_12_0: R[12,0] = 1 -> tau[12,0] - tau_star_12 = 0
 tau_star_eq_12_1: R[12,1] = 1 -> tau[12,1] - tau_star_12 = 0
 tau_star_eq_12_2: R[12,2] = 1 -> tau[12,2] - tau_star_12 = 0
 tau_star_eq_12_3: R[12,3] = 1 -> tau[12,3] - tau_star_12 = 0
 tau_star_eq_12_4: R[12,4] = 1 -> tau[12,4] - tau_star_12 = 0
 tau_star_eq_12_5: R[12,5] = 1 -> tau[12,5] - tau_star_12 = 0
 tau_star_eq_12_6: R[12,6] = 1 -> tau[12,6] - tau_star_12 = 0
 tau_star_eq_12_7: R[12,7] = 1 -> tau[12,7] - tau_star_12 = 0
 tau_star_eq_13_0: R[13,0] = 1 -> tau[13,0] - tau_star_13 = 0
 tau_star_eq_13_1: R[13,1] = 1 -> tau[13,1] - tau_star_13 = 0
 tau_star_eq_13_2: R[13,2] = 1 -> tau[13,2] - tau_star_13 = 0
 tau_star_eq_13_3: R[13,3] = 1 -> tau[13,3] - tau_star_13 = 0
 tau_star_eq_13_4: R[13,4] = 1 -> tau[13,4] - tau_star_13 = 0
 tau_star_eq_13_5: R[13,5] = 1 -> tau[13,5] - tau_star_13 = 0
 tau_star_eq_13_6: R[13,6] = 1 -> tau[13,6] - tau_star_13 = 0
 tau_star_eq_13_7: R[13,7] = 1 -> tau[13,7] - tau_star_13 = 0
 tau_star_eq_14_0: R[14,0] = 1 -> tau[14,0] - tau_star_14 = 0
 tau_star_eq_14_1: R[14,1] = 1 -> tau[14,1] - tau_star_14 = 0
 tau_star_eq_14_2: R[14,2] = 1 -> tau[14,2] - tau_star_14 = 0
 tau_star_eq_14_3: R[14,3] = 1 -> tau[14,3] - tau_star_14 = 0
 tau_star_eq_14_4: R[14,4] = 1 -> tau[14,4] - tau_star_14 = 0
 tau_star_eq_14_5: R[14,5] = 1 -> tau[14,5] - tau_star_14 = 0
 tau_star_eq_14_6: R[14,6] = 1 -> tau[14,6] - tau_star_14 = 0
 tau_star_eq_14_7: R[14,7] = 1 -> tau[14,7] - tau_star_14 = 0
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
 Q0_var[0] <= 2440.950183572415
 Q0_var[1] <= 2440.950183572415
 Q0_var[2] <= 2440.950183572415
 Q0_var[3] <= 2440.950183572415
 Q0_var[4] <= 2440.950183572415
 Q0_var[5] <= 2440.950183572415
 Q0_var[6] <= 2440.950183572415
 Q0_var[7] <= 2440.950183572415
 Q0_var[8] <= 2440.950183572415
 Q0_var[9] <= 2440.950183572415
 Q0_var[10] <= 2440.950183572415
 Q0_var[11] <= 2440.950183572415
 Q0_var[12] <= 2440.950183572415
 Q0_var[13] <= 2440.950183572415
 Q0_var[14] <= 2440.950183572415
 tau[0,0] free
 tau[0,1] free
 tau[0,2] free
 tau[0,3] free
 tau[0,4] free
 tau[0,5] free
 tau[0,6] free
 tau[0,7] free
 tau[1,0] free
 tau[1,1] free
 tau[1,2] free
 tau[1,3] free
 tau[1,4] free
 tau[1,5] free
 tau[1,6] free
 tau[1,7] free
 tau[2,0] free
 tau[2,1] free
 tau[2,2] free
 tau[2,3] free
 tau[2,4] free
 tau[2,5] free
 tau[2,6] free
 tau[2,7] free
 tau[3,0] free
 tau[3,1] free
 tau[3,2] free
 tau[3,3] free
 tau[3,4] free
 tau[3,5] free
 tau[3,6] free
 tau[3,7] free
 tau[4,0] free
 tau[4,1] free
 tau[4,2] free
 tau[4,3] free
 tau[4,4] free
 tau[4,5] free
 tau[4,6] free
 tau[4,7] free
 tau[5,0] free
 tau[5,1] free
 tau[5,2] free
 tau[5,3] free
 tau[5,4] free
 tau[5,5] free
 tau[5,6] free
 tau[5,7] free
 tau[6,0] free
 tau[6,1] free
 tau[6,2] free
 tau[6,3] free
 tau[6,4] free
 tau[6,5] free
 tau[6,6] free
 tau[6,7] free
 tau[7,0] free
 tau[7,1] free
 tau[7,2] free
 tau[7,3] free
 tau[7,4] free
 tau[7,5] free
 tau[7,6] free
 tau[7,7] free
 tau[8,0] free
 tau[8,1] free
 tau[8,2] free
 tau[8,3] free
 tau[8,4] free
 tau[8,5] free
 tau[8,6] free
 tau[8,7] free
 tau[9,0] free
 tau[9,1] free
 tau[9,2] free
 tau[9,3] free
 tau[9,4] free
 tau[9,5] free
 tau[9,6] free
 tau[9,7] free
 tau[10,0] free
 tau[10,1] free
 tau[10,2] free
 tau[10,3] free
 tau[10,4] free
 tau[10,5] free
 tau[10,6] free
 tau[10,7] free
 tau[11,0] free
 tau[11,1] free
 tau[11,2] free
 tau[11,3] free
 tau[11,4] free
 tau[11,5] free
 tau[11,6] free
 tau[11,7] free
 tau[12,0] free
 tau[12,1] free
 tau[12,2] free
 tau[12,3] free
 tau[12,4] free
 tau[12,5] free
 tau[12,6] free
 tau[12,7] free
 tau[13,0] free
 tau[13,1] free
 tau[13,2] free
 tau[13,3] free
 tau[13,4] free
 tau[13,5] free
 tau[13,6] free
 tau[13,7] free
 tau[14,0] free
 tau[14,1] free
 tau[14,2] free
 tau[14,3] free
 tau[14,4] free
 tau[14,5] free
 tau[14,6] free
 tau[14,7] free
 r[0,0] <= 1
 r[0,1] <= 1
 r[0,2] <= 1
 r[0,3] <= 1
 r[0,4] <= 1
 r[0,5] <= 1
 r[0,6] <= 1
 r[0,7] <= 1
 r[1,0] <= 1
 r[1,1] <= 1
 r[1,2] <= 1
 r[1,3] <= 1
 r[1,4] <= 1
 r[1,5] <= 1
 r[1,6] <= 1
 r[1,7] <= 1
 r[2,0] <= 1
 r[2,1] <= 1
 r[2,2] <= 1
 r[2,3] <= 1
 r[2,4] <= 1
 r[2,5] <= 1
 r[2,6] <= 1
 r[2,7] <= 1
 r[3,0] <= 1
 r[3,1] <= 1
 r[3,2] <= 1
 r[3,3] <= 1
 r[3,4] <= 1
 r[3,5] <= 1
 r[3,6] <= 1
 r[3,7] <= 1
 r[4,0] <= 1
 r[4,1] <= 1
 r[4,2] <= 1
 r[4,3] <= 1
 r[4,4] <= 1
 r[4,5] <= 1
 r[4,6] <= 1
 r[4,7] <= 1
 r[5,0] <= 1
 r[5,1] <= 1
 r[5,2] <= 1
 r[5,3] <= 1
 r[5,4] <= 1
 r[5,5] <= 1
 r[5,6] <= 1
 r[5,7] <= 1
 r[6,0] <= 1
 r[6,1] <= 1
 r[6,2] <= 1
 r[6,3] <= 1
 r[6,4] <= 1
 r[6,5] <= 1
 r[6,6] <= 1
 r[6,7] <= 1
 r[7,0] <= 1
 r[7,1] <= 1
 r[7,2] <= 1
 r[7,3] <= 1
 r[7,4] <= 1
 r[7,5] <= 1
 r[7,6] <= 1
 r[7,7] <= 1
 r[8,0] <= 1
 r[8,1] <= 1
 r[8,2] <= 1
 r[8,3] <= 1
 r[8,4] <= 1
 r[8,5] <= 1
 r[8,6] <= 1
 r[8,7] <= 1
 r[9,0] <= 1
 r[9,1] <= 1
 r[9,2] <= 1
 r[9,3] <= 1
 r[9,4] <= 1
 r[9,5] <= 1
 r[9,6] <= 1
 r[9,7] <= 1
 r[10,0] <= 1
 r[10,1] <= 1
 r[10,2] <= 1
 r[10,3] <= 1
 r[10,4] <= 1
 r[10,5] <= 1
 r[10,6] <= 1
 r[10,7] <= 1
 r[11,0] <= 1
 r[11,1] <= 1
 r[11,2] <= 1
 r[11,3] <= 1
 r[11,4] <= 1
 r[11,5] <= 1
 r[11,6] <= 1
 r[11,7] <= 1
 r[12,0] <= 1
 r[12,1] <= 1
 r[12,2] <= 1
 r[12,3] <= 1
 r[12,4] <= 1
 r[12,5] <= 1
 r[12,6] <= 1
 r[12,7] <= 1
 r[13,0] <= 1
 r[13,1] <= 1
 r[13,2] <= 1
 r[13,3] <= 1
 r[13,4] <= 1
 r[13,5] <= 1
 r[13,6] <= 1
 r[13,7] <= 1
 r[14,0] <= 1
 r[14,1] <= 1
 r[14,2] <= 1
 r[14,3] <= 1
 r[14,4] <= 1
 r[14,5] <= 1
 r[14,6] <= 1
 r[14,7] <= 1
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
 tau_star_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 tau_star_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 tau_star_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 tau_star_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 tau_star_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 tau_star_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 tau_star_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 tau_star_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 tau_star_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 tau_star_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 tau_star_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 tau_star_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 tau_star_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 tau_star_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 tau_star_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
Binaries
 R[0,0] R[0,1] R[0,2] R[0,3] R[0,4] R[0,5] R[0,6] R[0,7] R[1,0] R[1,1]
 R[1,2] R[1,3] R[1,4] R[1,5] R[1,6] R[1,7] R[2,0] R[2,1] R[2,2] R[2,3]
 R[2,4] R[2,5] R[2,6] R[2,7] R[3,0] R[3,1] R[3,2] R[3,3] R[3,4] R[3,5]
 R[3,6] R[3,7] R[4,0] R[4,1] R[4,2] R[4,3] R[4,4] R[4,5] R[4,6] R[4,7]
 R[5,0] R[5,1] R[5,2] R[5,3] R[5,4] R[5,5] R[5,6] R[5,7] R[6,0] R[6,1]
 R[6,2] R[6,3] R[6,4] R[6,5] R[6,6] R[6,7] R[7,0] R[7,1] R[7,2] R[7,3]
 R[7,4] R[7,5] R[7,6] R[7,7] R[8,0] R[8,1] R[8,2] R[8,3] R[8,4] R[8,5]
 R[8,6] R[8,7] R[9,0] R[9,1] R[9,2] R[9,3] R[9,4] R[9,5] R[9,6] R[9,7]
 R[10,0] R[10,1] R[10,2] R[10,3] R[10,4] R[10,5] R[10,6] R[10,7] R[11,0]
 R[11,1] R[11,2] R[11,3] R[11,4] R[11,5] R[11,6] R[11,7] R[12,0] R[12,1]
 R[12,2] R[12,3] R[12,4] R[12,5] R[12,6] R[12,7] R[13,0] R[13,1] R[13,2]
 R[13,3] R[13,4] R[13,5] R[13,6] R[13,7] R[14,0] R[14,1] R[14,2] R[14,3]
 R[14,4] R[14,5] R[14,6] R[14,7]
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
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 logistic_constraint_1: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[1] = LOGISTIC ( f_var[1] )
 Max_Constraint_1: Q1_var[1] = MAX ( Q_hat_adjusted[1] , 0 )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Total_Demand_Up_to_K_minus_1[1] ,
   Q0_var[1] )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Total_Demand_from_k_to_T[1] ,
   Q1_plus_left[1] )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 logistic_constraint_2: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[2] = LOGISTIC ( f_var[2] )
 Max_Constraint_2: Q1_var[2] = MAX ( Q_hat_adjusted[2] , 0 )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Total_Demand_Up_to_K_minus_1[2] ,
   Q0_var[2] )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Total_Demand_from_k_to_T[2] ,
   Q1_plus_left[2] )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 logistic_constraint_3: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[3] = LOGISTIC ( f_var[3] )
 Max_Constraint_3: Q1_var[3] = MAX ( Q_hat_adjusted[3] , 0 )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Total_Demand_Up_to_K_minus_1[3] ,
   Q0_var[3] )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Total_Demand_from_k_to_T[3] ,
   Q1_plus_left[3] )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 logistic_constraint_4: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[4] = LOGISTIC ( f_var[4] )
 Max_Constraint_4: Q1_var[4] = MAX ( Q_hat_adjusted[4] , 0 )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Total_Demand_Up_to_K_minus_1[4] ,
   Q0_var[4] )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Total_Demand_from_k_to_T[4] ,
   Q1_plus_left[4] )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 logistic_constraint_5: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[5] = LOGISTIC ( f_var[5] )
 Max_Constraint_5: Q1_var[5] = MAX ( Q_hat_adjusted[5] , 0 )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Total_Demand_Up_to_K_minus_1[5] ,
   Q0_var[5] )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Total_Demand_from_k_to_T[5] ,
   Q1_plus_left[5] )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 logistic_constraint_6: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[6] = LOGISTIC ( f_var[6] )
 Max_Constraint_6: Q1_var[6] = MAX ( Q_hat_adjusted[6] , 0 )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Total_Demand_Up_to_K_minus_1[6] ,
   Q0_var[6] )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Total_Demand_from_k_to_T[6] ,
   Q1_plus_left[6] )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 logistic_constraint_7: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[7] = LOGISTIC ( f_var[7] )
 Max_Constraint_7: Q1_var[7] = MAX ( Q_hat_adjusted[7] , 0 )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Total_Demand_Up_to_K_minus_1[7] ,
   Q0_var[7] )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Total_Demand_from_k_to_T[7] ,
   Q1_plus_left[7] )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 logistic_constraint_8: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[8] = LOGISTIC ( f_var[8] )
 Max_Constraint_8: Q1_var[8] = MAX ( Q_hat_adjusted[8] , 0 )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Total_Demand_Up_to_K_minus_1[8] ,
   Q0_var[8] )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Total_Demand_from_k_to_T[8] ,
   Q1_plus_left[8] )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 logistic_constraint_9: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[9] = LOGISTIC ( f_var[9] )
 Max_Constraint_9: Q1_var[9] = MAX ( Q_hat_adjusted[9] , 0 )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Total_Demand_Up_to_K_minus_1[9] ,
   Q0_var[9] )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Total_Demand_from_k_to_T[9] ,
   Q1_plus_left[9] )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 logistic_constraint_10: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[10] = LOGISTIC ( f_var[10] )
 Max_Constraint_10: Q1_var[10] = MAX ( Q_hat_adjusted[10] , 0 )
 Constr_Sold_0_min_10: Sold_0[10] = MIN (
   Total_Demand_Up_to_K_minus_1[10] , Q0_var[10] )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Total_Demand_from_k_to_T[10] ,
   Q1_plus_left[10] )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 logistic_constraint_11: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[11] = LOGISTIC ( f_var[11] )
 Max_Constraint_11: Q1_var[11] = MAX ( Q_hat_adjusted[11] , 0 )
 Constr_Sold_0_min_11: Sold_0[11] = MIN (
   Total_Demand_Up_to_K_minus_1[11] , Q0_var[11] )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Total_Demand_from_k_to_T[11] ,
   Q1_plus_left[11] )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 logistic_constraint_12: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[12] = LOGISTIC ( f_var[12] )
 Max_Constraint_12: Q1_var[12] = MAX ( Q_hat_adjusted[12] , 0 )
 Constr_Sold_0_min_12: Sold_0[12] = MIN (
   Total_Demand_Up_to_K_minus_1[12] , Q0_var[12] )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Total_Demand_from_k_to_T[12] ,
   Q1_plus_left[12] )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 logistic_constraint_13: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[13] = LOGISTIC ( f_var[13] )
 Max_Constraint_13: Q1_var[13] = MAX ( Q_hat_adjusted[13] , 0 )
 Constr_Sold_0_min_13: Sold_0[13] = MIN (
   Total_Demand_Up_to_K_minus_1[13] , Q0_var[13] )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Total_Demand_from_k_to_T[13] ,
   Q1_plus_left[13] )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 logistic_constraint_14: ( FuncNonlinear=1 )
   Fraction_for_second_order_amount[14] = LOGISTIC ( f_var[14] )
 Max_Constraint_14: Q1_var[14] = MAX ( Q_hat_adjusted[14] , 0 )
 Constr_Sold_0_min_14: Sold_0[14] = MIN (
   Total_Demand_Up_to_K_minus_1[14] , Q0_var[14] )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Total_Demand_from_k_to_T[14] ,
   Q1_plus_left[14] )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
End
