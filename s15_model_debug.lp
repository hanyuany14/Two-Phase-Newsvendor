\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - abs_beta[0,0] - abs_beta[0,1] - abs_beta[0,2] - abs_beta[0,3]
   - abs_beta[0,4] - abs_beta[1,0] - abs_beta[1,1] - abs_beta[1,2]
   - abs_beta[1,3] - abs_beta[1,4] - abs_beta[2,0] - abs_beta[2,1]
   - abs_beta[2,2] - abs_beta[2,3] - abs_beta[2,4] - abs_beta[3,0]
   - abs_beta[3,1] - abs_beta[3,2] - abs_beta[3,3] - abs_beta[3,4]
   - abs_beta[4,0] - abs_beta[4,1] - abs_beta[4,2] - abs_beta[4,3]
   - abs_beta[4,4] - abs_beta[5,0] - abs_beta[5,1] - abs_beta[5,2]
   - abs_beta[5,3] - abs_beta[5,4] - abs_beta[6,0] - abs_beta[6,1]
   - abs_beta[6,2] - abs_beta[6,3] - abs_beta[6,4] - abs_beta[7,0]
   - abs_beta[7,1] - abs_beta[7,2] - abs_beta[7,3] - abs_beta[7,4]
   + 0 Holding_Cost_0[0] + 0 Holding_Cost_0[1] + 0 Holding_Cost_0[2]
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
 R0: - betas[0,0] + abs_beta[0,0] >= 0
 R1: betas[0,0] + abs_beta[0,0] >= 0
 R2: - betas[0,1] + abs_beta[0,1] >= 0
 R3: betas[0,1] + abs_beta[0,1] >= 0
 R4: - betas[0,2] + abs_beta[0,2] >= 0
 R5: betas[0,2] + abs_beta[0,2] >= 0
 R6: - betas[0,3] + abs_beta[0,3] >= 0
 R7: betas[0,3] + abs_beta[0,3] >= 0
 R8: - betas[0,4] + abs_beta[0,4] >= 0
 R9: betas[0,4] + abs_beta[0,4] >= 0
 R10: - betas[1,0] + abs_beta[1,0] >= 0
 R11: betas[1,0] + abs_beta[1,0] >= 0
 R12: - betas[1,1] + abs_beta[1,1] >= 0
 R13: betas[1,1] + abs_beta[1,1] >= 0
 R14: - betas[1,2] + abs_beta[1,2] >= 0
 R15: betas[1,2] + abs_beta[1,2] >= 0
 R16: - betas[1,3] + abs_beta[1,3] >= 0
 R17: betas[1,3] + abs_beta[1,3] >= 0
 R18: - betas[1,4] + abs_beta[1,4] >= 0
 R19: betas[1,4] + abs_beta[1,4] >= 0
 R20: - betas[2,0] + abs_beta[2,0] >= 0
 R21: betas[2,0] + abs_beta[2,0] >= 0
 R22: - betas[2,1] + abs_beta[2,1] >= 0
 R23: betas[2,1] + abs_beta[2,1] >= 0
 R24: - betas[2,2] + abs_beta[2,2] >= 0
 R25: betas[2,2] + abs_beta[2,2] >= 0
 R26: - betas[2,3] + abs_beta[2,3] >= 0
 R27: betas[2,3] + abs_beta[2,3] >= 0
 R28: - betas[2,4] + abs_beta[2,4] >= 0
 R29: betas[2,4] + abs_beta[2,4] >= 0
 R30: - betas[3,0] + abs_beta[3,0] >= 0
 R31: betas[3,0] + abs_beta[3,0] >= 0
 R32: - betas[3,1] + abs_beta[3,1] >= 0
 R33: betas[3,1] + abs_beta[3,1] >= 0
 R34: - betas[3,2] + abs_beta[3,2] >= 0
 R35: betas[3,2] + abs_beta[3,2] >= 0
 R36: - betas[3,3] + abs_beta[3,3] >= 0
 R37: betas[3,3] + abs_beta[3,3] >= 0
 R38: - betas[3,4] + abs_beta[3,4] >= 0
 R39: betas[3,4] + abs_beta[3,4] >= 0
 R40: - betas[4,0] + abs_beta[4,0] >= 0
 R41: betas[4,0] + abs_beta[4,0] >= 0
 R42: - betas[4,1] + abs_beta[4,1] >= 0
 R43: betas[4,1] + abs_beta[4,1] >= 0
 R44: - betas[4,2] + abs_beta[4,2] >= 0
 R45: betas[4,2] + abs_beta[4,2] >= 0
 R46: - betas[4,3] + abs_beta[4,3] >= 0
 R47: betas[4,3] + abs_beta[4,3] >= 0
 R48: - betas[4,4] + abs_beta[4,4] >= 0
 R49: betas[4,4] + abs_beta[4,4] >= 0
 R50: - betas[5,0] + abs_beta[5,0] >= 0
 R51: betas[5,0] + abs_beta[5,0] >= 0
 R52: - betas[5,1] + abs_beta[5,1] >= 0
 R53: betas[5,1] + abs_beta[5,1] >= 0
 R54: - betas[5,2] + abs_beta[5,2] >= 0
 R55: betas[5,2] + abs_beta[5,2] >= 0
 R56: - betas[5,3] + abs_beta[5,3] >= 0
 R57: betas[5,3] + abs_beta[5,3] >= 0
 R58: - betas[5,4] + abs_beta[5,4] >= 0
 R59: betas[5,4] + abs_beta[5,4] >= 0
 R60: - betas[6,0] + abs_beta[6,0] >= 0
 R61: betas[6,0] + abs_beta[6,0] >= 0
 R62: - betas[6,1] + abs_beta[6,1] >= 0
 R63: betas[6,1] + abs_beta[6,1] >= 0
 R64: - betas[6,2] + abs_beta[6,2] >= 0
 R65: betas[6,2] + abs_beta[6,2] >= 0
 R66: - betas[6,3] + abs_beta[6,3] >= 0
 R67: betas[6,3] + abs_beta[6,3] >= 0
 R68: - betas[6,4] + abs_beta[6,4] >= 0
 R69: betas[6,4] + abs_beta[6,4] >= 0
 R70: - betas[7,0] + abs_beta[7,0] >= 0
 R71: betas[7,0] + abs_beta[7,0] >= 0
 R72: - betas[7,1] + abs_beta[7,1] >= 0
 R73: betas[7,1] + abs_beta[7,1] >= 0
 R74: - betas[7,2] + abs_beta[7,2] >= 0
 R75: betas[7,2] + abs_beta[7,2] >= 0
 R76: - betas[7,3] + abs_beta[7,3] >= 0
 R77: betas[7,3] + abs_beta[7,3] >= 0
 R78: - betas[7,4] + abs_beta[7,4] >= 0
 R79: betas[7,4] + abs_beta[7,4] >= 0
 R80: - 1.050344345027862 alphas[0] - 1.067548580664791 alphas[1]
   - 1.45845972753724 alphas[2] - 1.368778918491478 alphas[3] - alphas[4]
   + f_var[0] = 0
 Q0_upper_bound_0: - 2238.523560606579 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_0_0: - 1.050344345027862 betas[0,0]
   - 1.067548580664791 betas[0,1] - 1.45845972753724 betas[0,2]
   - 1.368778918491478 betas[0,3] - betas[0,4] + tau[0,0] = 0
 tau_computation_0_1: - 1.050344345027862 betas[1,0]
   - 1.067548580664791 betas[1,1] - 1.45845972753724 betas[1,2]
   - 1.368778918491478 betas[1,3] - betas[1,4] + tau[0,1] = 0
 tau_computation_0_2: - 1.050344345027862 betas[2,0]
   - 1.067548580664791 betas[2,1] - 1.45845972753724 betas[2,2]
   - 1.368778918491478 betas[2,3] - betas[2,4] + tau[0,2] = 0
 tau_computation_0_3: - 1.050344345027862 betas[3,0]
   - 1.067548580664791 betas[3,1] - 1.45845972753724 betas[3,2]
   - 1.368778918491478 betas[3,3] - betas[3,4] + tau[0,3] = 0
 tau_computation_0_4: - 1.050344345027862 betas[4,0]
   - 1.067548580664791 betas[4,1] - 1.45845972753724 betas[4,2]
   - 1.368778918491478 betas[4,3] - betas[4,4] + tau[0,4] = 0
 tau_computation_0_5: - 1.050344345027862 betas[5,0]
   - 1.067548580664791 betas[5,1] - 1.45845972753724 betas[5,2]
   - 1.368778918491478 betas[5,3] - betas[5,4] + tau[0,5] = 0
 tau_computation_0_6: - 1.050344345027862 betas[6,0]
   - 1.067548580664791 betas[6,1] - 1.45845972753724 betas[6,2]
   - 1.368778918491478 betas[6,3] - betas[6,4] + tau[0,6] = 0
 tau_computation_0_7: - 1.050344345027862 betas[7,0]
   - 1.067548580664791 betas[7,1] - 1.45845972753724 betas[7,2]
   - 1.368778918491478 betas[7,3] - betas[7,4] + tau[0,7] = 0
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
 Define_Q_hat_0: - 2468.933926345277 R[0,0] - 2968.558128856209 R[0,1]
   - 2867.75302975478 R[0,2] - 2841.448082161108 R[0,3]
   - 2836.699044184526 R[0,4] - 2836.699005858669 R[0,5]
   - 2807.55425342582 R[0,6] - 2788.157840197133 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 198.4847378639661 R[0,0] - 492.0487917072817 R[0,1]
   - 943.9702490147936 R[0,2] - 1237.534302569331 R[0,3]
   - 1520.427673984661 R[0,4] - 1813.991727189174 R[0,5]
   - 1877.811854496877 R[0,6] - 2201.028654573234 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 2589.672023118294 R[0,0] - 2296.107969274977 R[0,1]
   - 1844.186511967466 R[0,2] - 1550.622458412928 R[0,3]
   - 1267.729086997598 R[0,4] - 974.1650337930856 R[0,5]
   - 910.344906485382 R[0,6] - 587.1281064090251 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 700 Sold_0[0] + 700 Lost_0[0] - 700 Sold_1[0]
   + 300 Left_1[0] + 700 Lost_1[0] + profits_vars[0] = 0
 R109: - 0.0969168341474403 alphas[0] - 0.741640083241044 alphas[1]
   - 1.512703972516968 alphas[2] + 1.384929530718669 alphas[3] - alphas[4]
   + f_var[1] = 0
 Q0_upper_bound_1: - 2238.523560606579 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_1_0: - 0.0969168341474403 betas[0,0]
   - 0.741640083241044 betas[0,1] - 1.512703972516968 betas[0,2]
   + 1.384929530718669 betas[0,3] - betas[0,4] + tau[1,0] = 0
 tau_computation_1_1: - 0.0969168341474403 betas[1,0]
   - 0.741640083241044 betas[1,1] - 1.512703972516968 betas[1,2]
   + 1.384929530718669 betas[1,3] - betas[1,4] + tau[1,1] = 0
 tau_computation_1_2: - 0.0969168341474403 betas[2,0]
   - 0.741640083241044 betas[2,1] - 1.512703972516968 betas[2,2]
   + 1.384929530718669 betas[2,3] - betas[2,4] + tau[1,2] = 0
 tau_computation_1_3: - 0.0969168341474403 betas[3,0]
   - 0.741640083241044 betas[3,1] - 1.512703972516968 betas[3,2]
   + 1.384929530718669 betas[3,3] - betas[3,4] + tau[1,3] = 0
 tau_computation_1_4: - 0.0969168341474403 betas[4,0]
   - 0.741640083241044 betas[4,1] - 1.512703972516968 betas[4,2]
   + 1.384929530718669 betas[4,3] - betas[4,4] + tau[1,4] = 0
 tau_computation_1_5: - 0.0969168341474403 betas[5,0]
   - 0.741640083241044 betas[5,1] - 1.512703972516968 betas[5,2]
   + 1.384929530718669 betas[5,3] - betas[5,4] + tau[1,5] = 0
 tau_computation_1_6: - 0.0969168341474403 betas[6,0]
   - 0.741640083241044 betas[6,1] - 1.512703972516968 betas[6,2]
   + 1.384929530718669 betas[6,3] - betas[6,4] + tau[1,6] = 0
 tau_computation_1_7: - 0.0969168341474403 betas[7,0]
   - 0.741640083241044 betas[7,1] - 1.512703972516968 betas[7,2]
   + 1.384929530718669 betas[7,3] - betas[7,4] + tau[1,7] = 0
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
 Define_Q_hat_1: - 2866.586339535005 R[1,0] - 3058.052365289673 R[1,1]
   - 3112.773661446765 R[1,2] - 3113.800501918948 R[1,3]
   - 3013.909685770146 R[1,4] - 3013.909712384845 R[1,5]
   - 3060.205886935389 R[1,6] - 2970.3431191773 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 264.3486887195494 R[1,0] - 561.9349391059787 R[1,1]
   - 821.1361985923586 R[1,2] - 1118.722449173906 R[1,3]
   - 1500.5351176464 R[1,4] - 1798.121368036623 R[1,5]
   - 2129.663214778895 R[1,6] - 2375.16953935641 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 2705.993351417308 R[1,0] - 2408.407101030879 R[1,1]
   - 2149.205841544499 R[1,2] - 1851.619590962952 R[1,3]
   - 1469.806922490458 R[1,4] - 1172.220672100234 R[1,5]
   - 840.6788253579628 R[1,6] - 595.172500780447 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 700 Sold_0[1] + 700 Lost_0[1] - 700 Sold_1[1]
   + 300 Left_1[1] + 700 Lost_1[1] + profits_vars[1] = 0
 R138: - 2.033147710521636 alphas[0] + 0.9956786118187306 alphas[1]
   + 0.4471506084418529 alphas[2] + 0.9526297262402539 alphas[3]
   - alphas[4] + f_var[2] = 0
 Q0_upper_bound_2: - 2238.523560606579 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_2_0: - 2.033147710521636 betas[0,0]
   + 0.9956786118187306 betas[0,1] + 0.4471506084418529 betas[0,2]
   + 0.9526297262402539 betas[0,3] - betas[0,4] + tau[2,0] = 0
 tau_computation_2_1: - 2.033147710521636 betas[1,0]
   + 0.9956786118187306 betas[1,1] + 0.4471506084418529 betas[1,2]
   + 0.9526297262402539 betas[1,3] - betas[1,4] + tau[2,1] = 0
 tau_computation_2_2: - 2.033147710521636 betas[2,0]
   + 0.9956786118187306 betas[2,1] + 0.4471506084418529 betas[2,2]
   + 0.9526297262402539 betas[2,3] - betas[2,4] + tau[2,2] = 0
 tau_computation_2_3: - 2.033147710521636 betas[3,0]
   + 0.9956786118187306 betas[3,1] + 0.4471506084418529 betas[3,2]
   + 0.9526297262402539 betas[3,3] - betas[3,4] + tau[2,3] = 0
 tau_computation_2_4: - 2.033147710521636 betas[4,0]
   + 0.9956786118187306 betas[4,1] + 0.4471506084418529 betas[4,2]
   + 0.9526297262402539 betas[4,3] - betas[4,4] + tau[2,4] = 0
 tau_computation_2_5: - 2.033147710521636 betas[5,0]
   + 0.9956786118187306 betas[5,1] + 0.4471506084418529 betas[5,2]
   + 0.9526297262402539 betas[5,3] - betas[5,4] + tau[2,5] = 0
 tau_computation_2_6: - 2.033147710521636 betas[6,0]
   + 0.9956786118187306 betas[6,1] + 0.4471506084418529 betas[6,2]
   + 0.9526297262402539 betas[6,3] - betas[6,4] + tau[2,6] = 0
 tau_computation_2_7: - 2.033147710521636 betas[7,0]
   + 0.9956786118187306 betas[7,1] + 0.4471506084418529 betas[7,2]
   + 0.9526297262402539 betas[7,3] - betas[7,4] + tau[2,7] = 0
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
 Define_Q_hat_2: - 1349.004366761997 R[2,0] - 1485.721727847996 R[2,1]
   - 1455.039299896097 R[2,2] - 1447.807567306578 R[2,3]
   - 1335.233117868963 R[2,4] - 1335.23311666473 R[2,5]
   - 1299.692347370603 R[2,6] - 1208.791393724334 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 12.98860348262056 R[2,0] - 165.2516863549052 R[2,1]
   - 275.1646487743295 R[2,2] - 427.4281727672712 R[2,3]
   - 551.5670748175615 R[2,4] - 703.8305986608269 R[2,5]
   - 772.0211933668342 R[2,6] - 904.2632669070615 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1195.801711110972 R[2,0] - 1043.538628238687 R[2,1]
   - 933.6256658192627 R[2,2] - 781.362141826321 R[2,3]
   - 657.2232397760308 R[2,4] - 504.9597159327654 R[2,5]
   - 436.7691212267582 R[2,6] - 304.5270476865308 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 700 Sold_0[2] + 700 Lost_0[2] - 700 Sold_1[2]
   + 300 Left_1[2] + 700 Lost_1[2] + profits_vars[2] = 0
 R167: - 0.7043458392010258 alphas[0] - 1.016156428723332 alphas[1]
   + 1.276889314428171 alphas[2] + 0.2846885935058358 alphas[3] - alphas[4]
   + f_var[3] = 0
 Q0_upper_bound_3: - 2238.523560606579 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_3_0: - 0.7043458392010258 betas[0,0]
   - 1.016156428723332 betas[0,1] + 1.276889314428171 betas[0,2]
   + 0.2846885935058358 betas[0,3] - betas[0,4] + tau[3,0] = 0
 tau_computation_3_1: - 0.7043458392010258 betas[1,0]
   - 1.016156428723332 betas[1,1] + 1.276889314428171 betas[1,2]
   + 0.2846885935058358 betas[1,3] - betas[1,4] + tau[3,1] = 0
 tau_computation_3_2: - 0.7043458392010258 betas[2,0]
   - 1.016156428723332 betas[2,1] + 1.276889314428171 betas[2,2]
   + 0.2846885935058358 betas[2,3] - betas[2,4] + tau[3,2] = 0
 tau_computation_3_3: - 0.7043458392010258 betas[3,0]
   - 1.016156428723332 betas[3,1] + 1.276889314428171 betas[3,2]
   + 0.2846885935058358 betas[3,3] - betas[3,4] + tau[3,3] = 0
 tau_computation_3_4: - 0.7043458392010258 betas[4,0]
   - 1.016156428723332 betas[4,1] + 1.276889314428171 betas[4,2]
   + 0.2846885935058358 betas[4,3] - betas[4,4] + tau[3,4] = 0
 tau_computation_3_5: - 0.7043458392010258 betas[5,0]
   - 1.016156428723332 betas[5,1] + 1.276889314428171 betas[5,2]
   + 0.2846885935058358 betas[5,3] - betas[5,4] + tau[3,5] = 0
 tau_computation_3_6: - 0.7043458392010258 betas[6,0]
   - 1.016156428723332 betas[6,1] + 1.276889314428171 betas[6,2]
   + 0.2846885935058358 betas[6,3] - betas[6,4] + tau[3,6] = 0
 tau_computation_3_7: - 0.7043458392010258 betas[7,0]
   - 1.016156428723332 betas[7,1] + 1.276889314428171 betas[7,2]
   + 0.2846885935058358 betas[7,3] - betas[7,4] + tau[3,7] = 0
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
 Define_Q_hat_3: - 1683.361683644476 R[3,0] - 946.8347839778352 R[3,1]
   - 957.3925963073079 R[3,2] - 961.8017071248448 R[3,3]
   - 1221.963915620835 R[3,4] - 1221.963951913556 R[3,5]
   - 1223.90555941859 R[3,6] - 1138.599424756442 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 68.36886311867899 R[3,0] - 159.1602781707688 R[3,1]
   - 229.9756691062049 R[3,2] - 320.714275556847 R[3,3]
   - 604.2283229296429 R[3,4] - 694.9669294413246 R[3,5]
   - 904.7017339076951 R[3,6] - 957.1211326919937 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 1070.229482597832 R[3,0] - 979.4380675457419 R[3,1]
   - 908.6226766103058 R[3,2] - 817.8840701596636 R[3,3]
   - 534.3700227868677 R[3,4] - 443.6314162751861 R[3,5]
   - 233.8966118088155 R[3,6] - 181.477213024517 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 700 Sold_0[3] + 700 Lost_0[3] - 700 Sold_1[3]
   + 300 Left_1[3] + 700 Lost_1[3] + profits_vars[3] = 0
 R196: 0.2329247563597341 alphas[0] + 1.339231272668894 alphas[1]
   - 0.327321534244252 alphas[2] - 1.469610369037546 alphas[3] - alphas[4]
   + f_var[4] = 0
 Q0_upper_bound_4: - 2238.523560606579 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_4_0: 0.2329247563597341 betas[0,0]
   + 1.339231272668894 betas[0,1] - 0.327321534244252 betas[0,2]
   - 1.469610369037546 betas[0,3] - betas[0,4] + tau[4,0] = 0
 tau_computation_4_1: 0.2329247563597341 betas[1,0]
   + 1.339231272668894 betas[1,1] - 0.327321534244252 betas[1,2]
   - 1.469610369037546 betas[1,3] - betas[1,4] + tau[4,1] = 0
 tau_computation_4_2: 0.2329247563597341 betas[2,0]
   + 1.339231272668894 betas[2,1] - 0.327321534244252 betas[2,2]
   - 1.469610369037546 betas[2,3] - betas[2,4] + tau[4,2] = 0
 tau_computation_4_3: 0.2329247563597341 betas[3,0]
   + 1.339231272668894 betas[3,1] - 0.327321534244252 betas[3,2]
   - 1.469610369037546 betas[3,3] - betas[3,4] + tau[4,3] = 0
 tau_computation_4_4: 0.2329247563597341 betas[4,0]
   + 1.339231272668894 betas[4,1] - 0.327321534244252 betas[4,2]
   - 1.469610369037546 betas[4,3] - betas[4,4] + tau[4,4] = 0
 tau_computation_4_5: 0.2329247563597341 betas[5,0]
   + 1.339231272668894 betas[5,1] - 0.327321534244252 betas[5,2]
   - 1.469610369037546 betas[5,3] - betas[5,4] + tau[4,5] = 0
 tau_computation_4_6: 0.2329247563597341 betas[6,0]
   + 1.339231272668894 betas[6,1] - 0.327321534244252 betas[6,2]
   - 1.469610369037546 betas[6,3] - betas[6,4] + tau[4,6] = 0
 tau_computation_4_7: 0.2329247563597341 betas[7,0]
   + 1.339231272668894 betas[7,1] - 0.327321534244252 betas[7,2]
   - 1.469610369037546 betas[7,3] - betas[7,4] + tau[4,7] = 0
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
 Define_Q_hat_4: - 1463.633694482476 R[4,0] - 2044.352636708189 R[4,1]
   - 2104.383995628398 R[4,2] - 2091.003292936023 R[4,3]
   - 2091.236068852112 R[4,4] - 2091.23598240924 R[4,5]
   - 2079.926891601256 R[4,6] - 2036.822541020434 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 31.97488435799389 R[4,0] - 241.6653201327614 R[4,1]
   - 241.6653201327614 R[4,2] - 451.3557598893878 R[4,3]
   - 838.7009197935665 R[4,4] - 1048.391359135199 R[4,5]
   - 1363.070676980988 R[4,6] - 1617.440582841253 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 2004.846577166524 R[4,0] - 1795.156141391756 R[4,1]
   - 1795.156141391756 R[4,2] - 1585.46570163513 R[4,3]
   - 1198.120541730951 R[4,4] - 988.4301023893188 R[4,5]
   - 673.7507845435301 R[4,6] - 419.3808786832647 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 700 Sold_0[4] + 700 Lost_0[4] - 700 Sold_1[4]
   + 300 Left_1[4] + 700 Lost_1[4] + profits_vars[4] = 0
 R225: - 0.544268956839954 alphas[0] - 0.9766196684644459 alphas[1]
   + 0.1732075015909295 alphas[2] + 0.5623367486861119 alphas[3]
   - alphas[4] + f_var[5] = 0
 Q0_upper_bound_5: - 2238.523560606579 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_5_0: - 0.544268956839954 betas[0,0]
   - 0.9766196684644459 betas[0,1] + 0.1732075015909295 betas[0,2]
   + 0.5623367486861119 betas[0,3] - betas[0,4] + tau[5,0] = 0
 tau_computation_5_1: - 0.544268956839954 betas[1,0]
   - 0.9766196684644459 betas[1,1] + 0.1732075015909295 betas[1,2]
   + 0.5623367486861119 betas[1,3] - betas[1,4] + tau[5,1] = 0
 tau_computation_5_2: - 0.544268956839954 betas[2,0]
   - 0.9766196684644459 betas[2,1] + 0.1732075015909295 betas[2,2]
   + 0.5623367486861119 betas[2,3] - betas[2,4] + tau[5,2] = 0
 tau_computation_5_3: - 0.544268956839954 betas[3,0]
   - 0.9766196684644459 betas[3,1] + 0.1732075015909295 betas[3,2]
   + 0.5623367486861119 betas[3,3] - betas[3,4] + tau[5,3] = 0
 tau_computation_5_4: - 0.544268956839954 betas[4,0]
   - 0.9766196684644459 betas[4,1] + 0.1732075015909295 betas[4,2]
   + 0.5623367486861119 betas[4,3] - betas[4,4] + tau[5,4] = 0
 tau_computation_5_5: - 0.544268956839954 betas[5,0]
   - 0.9766196684644459 betas[5,1] + 0.1732075015909295 betas[5,2]
   + 0.5623367486861119 betas[5,3] - betas[5,4] + tau[5,5] = 0
 tau_computation_5_6: - 0.544268956839954 betas[6,0]
   - 0.9766196684644459 betas[6,1] + 0.1732075015909295 betas[6,2]
   + 0.5623367486861119 betas[6,3] - betas[6,4] + tau[5,6] = 0
 tau_computation_5_7: - 0.544268956839954 betas[7,0]
   - 0.9766196684644459 betas[7,1] + 0.1732075015909295 betas[7,2]
   + 0.5623367486861119 betas[7,3] - betas[7,4] + tau[5,7] = 0
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
 Define_Q_hat_5: - 2363.129427258456 R[5,0] - 1809.183444588584 R[5,1]
   - 1809.048077746383 R[5,2] - 1829.685209511513 R[5,3]
   - 1713.041399235066 R[5,4] - 1713.041364884913 R[5,5]
   - 1694.265876447782 R[5,6] - 1539.177212197813 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 180.9601307020097 R[5,0] - 353.5369765402268 R[5,1]
   - 578.0171477195959 R[5,2] - 750.5934849419162 R[5,3]
   - 880.4778811493943 R[5,4] - 1053.054217786579 R[5,5]
   - 1132.615229539439 R[5,6] - 1194.023459495702 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 1358.216002068061 R[5,0] - 1185.639156229844 R[5,1]
   - 961.1589850504745 R[5,2] - 788.5826478281542 R[5,3]
   - 658.698251620676 R[5,4] - 486.1219149834918 R[5,5]
   - 406.560903230631 R[5,6] - 345.1526732743686 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 700 Sold_0[5] + 700 Lost_0[5] - 700 Sold_1[5]
   + 300 Left_1[5] + 700 Lost_1[5] + profits_vars[5] = 0
 R254: 1.955855656884161 alphas[0] - 0.9509925722571744 alphas[1]
   - 1.509550811917107 alphas[2] + 0.9972012482919431 alphas[3] - alphas[4]
   + f_var[6] = 0
 Q0_upper_bound_6: - 2238.523560606579 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_6_0: 1.955855656884161 betas[0,0]
   - 0.9509925722571744 betas[0,1] - 1.509550811917107 betas[0,2]
   + 0.9972012482919431 betas[0,3] - betas[0,4] + tau[6,0] = 0
 tau_computation_6_1: 1.955855656884161 betas[1,0]
   - 0.9509925722571744 betas[1,1] - 1.509550811917107 betas[1,2]
   + 0.9972012482919431 betas[1,3] - betas[1,4] + tau[6,1] = 0
 tau_computation_6_2: 1.955855656884161 betas[2,0]
   - 0.9509925722571744 betas[2,1] - 1.509550811917107 betas[2,2]
   + 0.9972012482919431 betas[2,3] - betas[2,4] + tau[6,2] = 0
 tau_computation_6_3: 1.955855656884161 betas[3,0]
   - 0.9509925722571744 betas[3,1] - 1.509550811917107 betas[3,2]
   + 0.9972012482919431 betas[3,3] - betas[3,4] + tau[6,3] = 0
 tau_computation_6_4: 1.955855656884161 betas[4,0]
   - 0.9509925722571744 betas[4,1] - 1.509550811917107 betas[4,2]
   + 0.9972012482919431 betas[4,3] - betas[4,4] + tau[6,4] = 0
 tau_computation_6_5: 1.955855656884161 betas[5,0]
   - 0.9509925722571744 betas[5,1] - 1.509550811917107 betas[5,2]
   + 0.9972012482919431 betas[5,3] - betas[5,4] + tau[6,5] = 0
 tau_computation_6_6: 1.955855656884161 betas[6,0]
   - 0.9509925722571744 betas[6,1] - 1.509550811917107 betas[6,2]
   + 0.9972012482919431 betas[6,3] - betas[6,4] + tau[6,6] = 0
 tau_computation_6_7: 1.955855656884161 betas[7,0]
   - 0.9509925722571744 betas[7,1] - 1.509550811917107 betas[7,2]
   + 0.9972012482919431 betas[7,3] - betas[7,4] + tau[6,7] = 0
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
 Define_Q_hat_6: - 3175.2914114999 R[6,0] - 3095.745579969062 R[6,1]
   - 3124.438870596113 R[6,2] - 3132.52351106542 R[6,3]
   - 3241.580355017733 R[6,4] - 3241.580371115428 R[6,5]
   - 3174.840088229079 R[6,6] - 3184.990406528357 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 315.480116605702 R[6,0] - 612.8325606350786 R[6,1]
   - 962.6182824774246 R[6,2] - 1259.970727083951 R[6,3]
   - 1725.680741051027 R[6,4] - 2023.033185372135 R[6,5]
   - 2216.844655169977 R[6,6] - 2590.284438606934 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 2869.509210643447 R[6,0] - 2572.15676661407 R[6,1]
   - 2222.371044771724 R[6,2] - 1925.018600165198 R[6,3]
   - 1459.308586198121 R[6,4] - 1161.956141877014 R[6,5]
   - 968.1446720791719 R[6,6] - 594.7048886422157 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 700 Sold_0[6] + 700 Lost_0[6] - 700 Sold_1[6]
   + 300 Left_1[6] + 700 Lost_1[6] + profits_vars[6] = 0
 R283: 0.7535579465308403 alphas[0] - 0.4921736069494833 alphas[1]
   + 1.6061224519079 alphas[2] - 1.264463043062054 alphas[3] - alphas[4]
   + f_var[7] = 0
 Q0_upper_bound_7: - 2238.523560606579 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_7_0: 0.7535579465308403 betas[0,0]
   - 0.4921736069494833 betas[0,1] + 1.6061224519079 betas[0,2]
   - 1.264463043062054 betas[0,3] - betas[0,4] + tau[7,0] = 0
 tau_computation_7_1: 0.7535579465308403 betas[1,0]
   - 0.4921736069494833 betas[1,1] + 1.6061224519079 betas[1,2]
   - 1.264463043062054 betas[1,3] - betas[1,4] + tau[7,1] = 0
 tau_computation_7_2: 0.7535579465308403 betas[2,0]
   - 0.4921736069494833 betas[2,1] + 1.6061224519079 betas[2,2]
   - 1.264463043062054 betas[2,3] - betas[2,4] + tau[7,2] = 0
 tau_computation_7_3: 0.7535579465308403 betas[3,0]
   - 0.4921736069494833 betas[3,1] + 1.6061224519079 betas[3,2]
   - 1.264463043062054 betas[3,3] - betas[3,4] + tau[7,3] = 0
 tau_computation_7_4: 0.7535579465308403 betas[4,0]
   - 0.4921736069494833 betas[4,1] + 1.6061224519079 betas[4,2]
   - 1.264463043062054 betas[4,3] - betas[4,4] + tau[7,4] = 0
 tau_computation_7_5: 0.7535579465308403 betas[5,0]
   - 0.4921736069494833 betas[5,1] + 1.6061224519079 betas[5,2]
   - 1.264463043062054 betas[5,3] - betas[5,4] + tau[7,5] = 0
 tau_computation_7_6: 0.7535579465308403 betas[6,0]
   - 0.4921736069494833 betas[6,1] + 1.6061224519079 betas[6,2]
   - 1.264463043062054 betas[6,3] - betas[6,4] + tau[7,6] = 0
 tau_computation_7_7: 0.7535579465308403 betas[7,0]
   - 0.4921736069494833 betas[7,1] + 1.6061224519079 betas[7,2]
   - 1.264463043062054 betas[7,3] - betas[7,4] + tau[7,7] = 0
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
 Define_Q_hat_7: - 2264.705121439347 R[7,0] - 792.9074615224337 R[7,1]
   - 813.3791295919689 R[7,2] - 747.4467834804848 R[7,3]
   - 660.956848973449 R[7,4] - 660.9567054342471 R[7,5]
   - 653.1819773502729 R[7,6] - 632.0993270283326 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 164.6579194581119 R[7,0] - 231.2582303431065 R[7,1]
   - 366.7940402727327 R[7,2] - 433.1200924204296 R[7,3]
   - 433.1200924204296 R[7,4] - 499.4461442083728 R[7,5]
   - 499.4461442083728 R[7,6] - 499.4461442083728 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 467.4403282747305 R[7,0] - 400.8400173897359 R[7,1]
   - 265.3042074601098 R[7,2] - 198.9781553124128 R[7,3]
   - 198.9781553124128 R[7,4] - 132.6521035244696 R[7,5]
   - 132.6521035244696 R[7,6] - 132.6521035244696 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 700 Sold_0[7] + 700 Lost_0[7] - 700 Sold_1[7]
   + 300 Left_1[7] + 700 Lost_1[7] + profits_vars[7] = 0
 R312: - 0.7732497241358117 alphas[0] + 0.0524212248861011 alphas[1]
   - 0.8144192035245688 alphas[2] + 0.106270709504071 alphas[3] - alphas[4]
   + f_var[8] = 0
 Q0_upper_bound_8: - 2238.523560606579 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_8_0: - 0.7732497241358117 betas[0,0]
   + 0.0524212248861011 betas[0,1] - 0.8144192035245688 betas[0,2]
   + 0.106270709504071 betas[0,3] - betas[0,4] + tau[8,0] = 0
 tau_computation_8_1: - 0.7732497241358117 betas[1,0]
   + 0.0524212248861011 betas[1,1] - 0.8144192035245688 betas[1,2]
   + 0.106270709504071 betas[1,3] - betas[1,4] + tau[8,1] = 0
 tau_computation_8_2: - 0.7732497241358117 betas[2,0]
   + 0.0524212248861011 betas[2,1] - 0.8144192035245688 betas[2,2]
   + 0.106270709504071 betas[2,3] - betas[2,4] + tau[8,2] = 0
 tau_computation_8_3: - 0.7732497241358117 betas[3,0]
   + 0.0524212248861011 betas[3,1] - 0.8144192035245688 betas[3,2]
   + 0.106270709504071 betas[3,3] - betas[3,4] + tau[8,3] = 0
 tau_computation_8_4: - 0.7732497241358117 betas[4,0]
   + 0.0524212248861011 betas[4,1] - 0.8144192035245688 betas[4,2]
   + 0.106270709504071 betas[4,3] - betas[4,4] + tau[8,4] = 0
 tau_computation_8_5: - 0.7732497241358117 betas[5,0]
   + 0.0524212248861011 betas[5,1] - 0.8144192035245688 betas[5,2]
   + 0.106270709504071 betas[5,3] - betas[5,4] + tau[8,5] = 0
 tau_computation_8_6: - 0.7732497241358117 betas[6,0]
   + 0.0524212248861011 betas[6,1] - 0.8144192035245688 betas[6,2]
   + 0.106270709504071 betas[6,3] - betas[6,4] + tau[8,6] = 0
 tau_computation_8_7: - 0.7732497241358117 betas[7,0]
   + 0.0524212248861011 betas[7,1] - 0.8144192035245688 betas[7,2]
   + 0.106270709504071 betas[7,3] - betas[7,4] + tau[8,7] = 0
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
 Define_Q_hat_8: - 2315.004555093732 R[8,0] - 2496.435614381303 R[8,1]
   - 2472.148739695301 R[8,2] - 2464.768174771687 R[8,3]
   - 2411.206701731378 R[8,4] - 2411.206551062503 R[8,5]
   - 2413.847026888267 R[8,6] - 2345.685202338952 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 172.9891135270772 R[8,0] - 418.7977234178429 R[8,1]
   - 703.0444004931053 R[8,2] - 948.8530109994794 R[8,3]
   - 1216.687247942525 R[8,4] - 1462.495857520978 R[8,5]
   - 1628.377882892968 R[8,6] - 1854.066903382031 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 2172.695009011861 R[8,0] - 1926.886399121095 R[8,1]
   - 1642.639722045833 R[8,2] - 1396.831111539459 R[8,3]
   - 1128.996874596413 R[8,4] - 883.1882650179598 R[8,5]
   - 717.3062396459707 R[8,6] - 491.6172191569066 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 700 Sold_0[8] + 700 Lost_0[8] - 700 Sold_1[8]
   + 300 Left_1[8] + 700 Lost_1[8] + profits_vars[8] = 0
 R341: 0.8025096760880669 alphas[0] - 0.7344846151532984 alphas[1]
   + 0.8539476596650883 alphas[2] - 1.331676751790723 alphas[3] - alphas[4]
   + f_var[9] = 0
 Q0_upper_bound_9: - 2238.523560606579 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_9_0: 0.8025096760880669 betas[0,0]
   - 0.7344846151532984 betas[0,1] + 0.8539476596650883 betas[0,2]
   - 1.331676751790723 betas[0,3] - betas[0,4] + tau[9,0] = 0
 tau_computation_9_1: 0.8025096760880669 betas[1,0]
   - 0.7344846151532984 betas[1,1] + 0.8539476596650883 betas[1,2]
   - 1.331676751790723 betas[1,3] - betas[1,4] + tau[9,1] = 0
 tau_computation_9_2: 0.8025096760880669 betas[2,0]
   - 0.7344846151532984 betas[2,1] + 0.8539476596650883 betas[2,2]
   - 1.331676751790723 betas[2,3] - betas[2,4] + tau[9,2] = 0
 tau_computation_9_3: 0.8025096760880669 betas[3,0]
   - 0.7344846151532984 betas[3,1] + 0.8539476596650883 betas[3,2]
   - 1.331676751790723 betas[3,3] - betas[3,4] + tau[9,3] = 0
 tau_computation_9_4: 0.8025096760880669 betas[4,0]
   - 0.7344846151532984 betas[4,1] + 0.8539476596650883 betas[4,2]
   - 1.331676751790723 betas[4,3] - betas[4,4] + tau[9,4] = 0
 tau_computation_9_5: 0.8025096760880669 betas[5,0]
   - 0.7344846151532984 betas[5,1] + 0.8539476596650883 betas[5,2]
   - 1.331676751790723 betas[5,3] - betas[5,4] + tau[9,5] = 0
 tau_computation_9_6: 0.8025096760880669 betas[6,0]
   - 0.7344846151532984 betas[6,1] + 0.8539476596650883 betas[6,2]
   - 1.331676751790723 betas[6,3] - betas[6,4] + tau[9,6] = 0
 tau_computation_9_7: 0.8025096760880669 betas[7,0]
   - 0.7344846151532984 betas[7,1] + 0.8539476596650883 betas[7,2]
   - 1.331676751790723 betas[7,3] - betas[7,4] + tau[9,7] = 0
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
 Define_Q_hat_9: - 1719.981460728389 R[9,0] - 1248.063511052949 R[9,1]
   - 1286.146294312436 R[9,2] - 1306.587063656741 R[9,3]
   - 1162.453805231356 R[9,4] - 1162.453855651223 R[9,5]
   - 1115.793389734912 R[9,6] - 1172.257489265448 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 74.43426877999562 R[9,0] - 196.5359471421425 R[9,1]
   - 239.4985576768833 R[9,2] - 361.5981821063094 R[9,3]
   - 471.0724632845839 R[9,4] - 593.1720876174574 R[9,5]
   - 680.9855869456663 R[9,6] - 928.0571612720039 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 1097.822141157829 R[9,0] - 975.7204627956819 R[9,1]
   - 932.757852260941 R[9,2] - 810.658227831515 R[9,3]
   - 701.1839466532406 R[9,4] - 579.0843223203671 R[9,5]
   - 491.2708229921581 R[9,6] - 244.1992486658206 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 700 Sold_0[9] + 700 Lost_0[9] - 700 Sold_1[9]
   + 300 Left_1[9] + 700 Lost_1[9] + profits_vars[9] = 0
 R370: - 0.5937529225790621 alphas[0] + 1.141347397079515 alphas[1]
   + 1.012342925565199 alphas[2] + 0.9979402792570548 alphas[3] - alphas[4]
   + f_var[10] = 0
 Q0_upper_bound_10:
   - 2238.523560606579 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 tau_computation_10_0: - 0.5937529225790621 betas[0,0]
   + 1.141347397079515 betas[0,1] + 1.012342925565199 betas[0,2]
   + 0.9979402792570548 betas[0,3] - betas[0,4] + tau[10,0] = 0
 tau_computation_10_1: - 0.5937529225790621 betas[1,0]
   + 1.141347397079515 betas[1,1] + 1.012342925565199 betas[1,2]
   + 0.9979402792570548 betas[1,3] - betas[1,4] + tau[10,1] = 0
 tau_computation_10_2: - 0.5937529225790621 betas[2,0]
   + 1.141347397079515 betas[2,1] + 1.012342925565199 betas[2,2]
   + 0.9979402792570548 betas[2,3] - betas[2,4] + tau[10,2] = 0
 tau_computation_10_3: - 0.5937529225790621 betas[3,0]
   + 1.141347397079515 betas[3,1] + 1.012342925565199 betas[3,2]
   + 0.9979402792570548 betas[3,3] - betas[3,4] + tau[10,3] = 0
 tau_computation_10_4: - 0.5937529225790621 betas[4,0]
   + 1.141347397079515 betas[4,1] + 1.012342925565199 betas[4,2]
   + 0.9979402792570548 betas[4,3] - betas[4,4] + tau[10,4] = 0
 tau_computation_10_5: - 0.5937529225790621 betas[5,0]
   + 1.141347397079515 betas[5,1] + 1.012342925565199 betas[5,2]
   + 0.9979402792570548 betas[5,3] - betas[5,4] + tau[10,5] = 0
 tau_computation_10_6: - 0.5937529225790621 betas[6,0]
   + 1.141347397079515 betas[6,1] + 1.012342925565199 betas[6,2]
   + 0.9979402792570548 betas[6,3] - betas[6,4] + tau[10,6] = 0
 tau_computation_10_7: - 0.5937529225790621 betas[7,0]
   + 1.141347397079515 betas[7,1] + 1.012342925565199 betas[7,2]
   + 0.9979402792570548 betas[7,3] - betas[7,4] + tau[10,7] = 0
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
 Define_Q_hat_10: - 2228.083903964372 R[10,0] - 1202.721156914823 R[10,1]
   - 1146.905829571441 R[10,2] - 1168.393422075013 R[10,3]
   - 1100.538960475414 R[10,4] - 1100.539019691571 R[10,5]
   - 1114.271186076258 R[10,6] - 1065.357927097103 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 158.5922752219921 R[10,0] - 268.9694176567522 R[10,1]
   - 540.769271148211 R[10,2] - 651.1239263314422 R[10,3]
   - 652.7361806947544 R[10,4] - 763.0908357138978 R[10,5]
   - 767.5711985150378 R[10,6] - 844.6475378978021 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 906.764572714097 R[10,0] - 796.3874302793369 R[10,1]
   - 524.5875767878779 R[10,2] - 414.2329216046467 R[10,3]
   - 412.6206672413347 R[10,4] - 302.2660122221912 R[10,5]
   - 297.7856494210512 R[10,6] - 220.709310038287 R[10,7] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: - Total_Demand_Up_to_K_minus_1[10] + Q0_var[10]
   + Lost_0_aux_10 = 0
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: - Total_Demand_from_k_to_T[10]
   + Q1_plus_left[10] + Lost_0_aux_10 = 0
 Profit_Constraint_10: - 700 Sold_0[10] + 700 Lost_0[10] - 700 Sold_1[10]
   + 300 Left_1[10] + 700 Lost_1[10] + profits_vars[10] = 0
 R399: 0.2120038725951724 alphas[0] - 0.6526946592810959 alphas[1]
   - 0.4073717511202417 alphas[2] - 0.5627934797780342 alphas[3]
   - alphas[4] + f_var[11] = 0
 Q0_upper_bound_11:
   - 2238.523560606579 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 tau_computation_11_0: 0.2120038725951724 betas[0,0]
   - 0.6526946592810959 betas[0,1] - 0.4073717511202417 betas[0,2]
   - 0.5627934797780342 betas[0,3] - betas[0,4] + tau[11,0] = 0
 tau_computation_11_1: 0.2120038725951724 betas[1,0]
   - 0.6526946592810959 betas[1,1] - 0.4073717511202417 betas[1,2]
   - 0.5627934797780342 betas[1,3] - betas[1,4] + tau[11,1] = 0
 tau_computation_11_2: 0.2120038725951724 betas[2,0]
   - 0.6526946592810959 betas[2,1] - 0.4073717511202417 betas[2,2]
   - 0.5627934797780342 betas[2,3] - betas[2,4] + tau[11,2] = 0
 tau_computation_11_3: 0.2120038725951724 betas[3,0]
   - 0.6526946592810959 betas[3,1] - 0.4073717511202417 betas[3,2]
   - 0.5627934797780342 betas[3,3] - betas[3,4] + tau[11,3] = 0
 tau_computation_11_4: 0.2120038725951724 betas[4,0]
   - 0.6526946592810959 betas[4,1] - 0.4073717511202417 betas[4,2]
   - 0.5627934797780342 betas[4,3] - betas[4,4] + tau[11,4] = 0
 tau_computation_11_5: 0.2120038725951724 betas[5,0]
   - 0.6526946592810959 betas[5,1] - 0.4073717511202417 betas[5,2]
   - 0.5627934797780342 betas[5,3] - betas[5,4] + tau[11,5] = 0
 tau_computation_11_6: 0.2120038725951724 betas[6,0]
   - 0.6526946592810959 betas[6,1] - 0.4073717511202417 betas[6,2]
   - 0.5627934797780342 betas[6,3] - betas[6,4] + tau[11,6] = 0
 tau_computation_11_7: 0.2120038725951724 betas[7,0]
   - 0.6526946592810959 betas[7,1] - 0.4073717511202417 betas[7,2]
   - 0.5627934797780342 betas[7,3] - betas[7,4] + tau[11,7] = 0
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
 Define_Q_hat_11: - 2142.116079576946 R[11,0] - 2188.270726031536 R[11,1]
   - 2183.227734772508 R[11,2] - 2182.406707693785 R[11,3]
   - 2140.735731405577 R[11,4] - 2140.735706624867 R[11,5]
   - 2202.316532380587 R[11,6] - 2246.730941722328 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 144.3532554831296 R[11,0] - 359.9793949585815 R[11,1]
   - 574.4363992513064 R[11,2] - 790.0625424909911 R[11,3]
   - 1046.944633198257 R[11,4] - 1262.570776081426 R[11,5]
   - 1520.551427160388 R[11,6] - 1815.47757650403 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 2102.376606787238 R[11,0] - 1886.750467311786 R[11,1]
   - 1672.293463019062 R[11,2] - 1456.667319779377 R[11,3]
   - 1199.785229072111 R[11,4] - 984.1590861889422 R[11,5]
   - 726.1784351099795 R[11,6] - 431.252285766338 R[11,7] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: - Total_Demand_Up_to_K_minus_1[11] + Q0_var[11]
   + Lost_0_aux_11 = 0
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: - Total_Demand_from_k_to_T[11]
   + Q1_plus_left[11] + Lost_0_aux_11 = 0
 Profit_Constraint_11: - 700 Sold_0[11] + 700 Lost_0[11] - 700 Sold_1[11]
   + 300 Left_1[11] + 700 Lost_1[11] + profits_vars[11] = 0
 R428: 1.460765998943228 alphas[0] + 0.0929903885175016 alphas[1]
   + 0.9966835169504571 alphas[2] + 0.1253734837938557 alphas[3]
   - alphas[4] + f_var[12] = 0
 Q0_upper_bound_12:
   - 2238.523560606579 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 tau_computation_12_0: 1.460765998943228 betas[0,0]
   + 0.0929903885175016 betas[0,1] + 0.9966835169504571 betas[0,2]
   + 0.1253734837938557 betas[0,3] - betas[0,4] + tau[12,0] = 0
 tau_computation_12_1: 1.460765998943228 betas[1,0]
   + 0.0929903885175016 betas[1,1] + 0.9966835169504571 betas[1,2]
   + 0.1253734837938557 betas[1,3] - betas[1,4] + tau[12,1] = 0
 tau_computation_12_2: 1.460765998943228 betas[2,0]
   + 0.0929903885175016 betas[2,1] + 0.9966835169504571 betas[2,2]
   + 0.1253734837938557 betas[2,3] - betas[2,4] + tau[12,2] = 0
 tau_computation_12_3: 1.460765998943228 betas[3,0]
   + 0.0929903885175016 betas[3,1] + 0.9966835169504571 betas[3,2]
   + 0.1253734837938557 betas[3,3] - betas[3,4] + tau[12,3] = 0
 tau_computation_12_4: 1.460765998943228 betas[4,0]
   + 0.0929903885175016 betas[4,1] + 0.9966835169504571 betas[4,2]
   + 0.1253734837938557 betas[4,3] - betas[4,4] + tau[12,4] = 0
 tau_computation_12_5: 1.460765998943228 betas[5,0]
   + 0.0929903885175016 betas[5,1] + 0.9966835169504571 betas[5,2]
   + 0.1253734837938557 betas[5,3] - betas[5,4] + tau[12,5] = 0
 tau_computation_12_6: 1.460765998943228 betas[6,0]
   + 0.0929903885175016 betas[6,1] + 0.9966835169504571 betas[6,2]
   + 0.1253734837938557 betas[6,3] - betas[6,4] + tau[12,6] = 0
 tau_computation_12_7: 1.460765998943228 betas[7,0]
   + 0.0929903885175016 betas[7,1] + 0.9966835169504571 betas[7,2]
   + 0.1253734837938557 betas[7,3] - betas[7,4] + tau[12,7] = 0
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
 Define_Q_hat_12: - 1970.453961342662 R[12,0] - 1180.274404247524 R[12,1]
   - 1144.850837962175 R[12,2] - 1166.402907262423 R[12,3]
   - 1201.709759126568 R[12,4] - 1201.709723182952 R[12,5]
   - 1216.877198750338 R[12,6] - 1235.894237978137 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 115.9205214065135 R[12,0] - 227.4442698879841 R[12,1]
   - 426.1919903080167 R[12,2] - 537.7077870889788 R[12,3]
   - 644.8517192132558 R[12,4] - 756.367515460997 R[12,5]
   - 848.0092501629586 R[12,6] - 1012.861565764827 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 1119.972636853794 R[12,0] - 1008.448888372323 R[12,1]
   - 809.7011679522907 R[12,2] - 698.1853711713286 R[12,3]
   - 591.0414390470515 R[12,4] - 479.5256427993104 R[12,5]
   - 387.8839080973488 R[12,6] - 223.0315924954805 R[12,7] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: - Total_Demand_Up_to_K_minus_1[12] + Q0_var[12]
   + Lost_0_aux_12 = 0
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: - Total_Demand_from_k_to_T[12]
   + Q1_plus_left[12] + Lost_0_aux_12 = 0
 Profit_Constraint_12: - 700 Sold_0[12] + 700 Lost_0[12] - 700 Sold_1[12]
   + 300 Left_1[12] + 700 Lost_1[12] + profits_vars[12] = 0
 R457: 0.7675764016808295 alphas[0] + 2.044247600435395 alphas[1]
   - 0.4186817893938694 alphas[2] + 1.258880578516843 alphas[3] - alphas[4]
   + f_var[13] = 0
 Q0_upper_bound_13:
   - 2238.523560606579 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 tau_computation_13_0: 0.7675764016808295 betas[0,0]
   + 2.044247600435395 betas[0,1] - 0.4186817893938694 betas[0,2]
   + 1.258880578516843 betas[0,3] - betas[0,4] + tau[13,0] = 0
 tau_computation_13_1: 0.7675764016808295 betas[1,0]
   + 2.044247600435395 betas[1,1] - 0.4186817893938694 betas[1,2]
   + 1.258880578516843 betas[1,3] - betas[1,4] + tau[13,1] = 0
 tau_computation_13_2: 0.7675764016808295 betas[2,0]
   + 2.044247600435395 betas[2,1] - 0.4186817893938694 betas[2,2]
   + 1.258880578516843 betas[2,3] - betas[2,4] + tau[13,2] = 0
 tau_computation_13_3: 0.7675764016808295 betas[3,0]
   + 2.044247600435395 betas[3,1] - 0.4186817893938694 betas[3,2]
   + 1.258880578516843 betas[3,3] - betas[3,4] + tau[13,3] = 0
 tau_computation_13_4: 0.7675764016808295 betas[4,0]
   + 2.044247600435395 betas[4,1] - 0.4186817893938694 betas[4,2]
   + 1.258880578516843 betas[4,3] - betas[4,4] + tau[13,4] = 0
 tau_computation_13_5: 0.7675764016808295 betas[5,0]
   + 2.044247600435395 betas[5,1] - 0.4186817893938694 betas[5,2]
   + 1.258880578516843 betas[5,3] - betas[5,4] + tau[13,5] = 0
 tau_computation_13_6: 0.7675764016808295 betas[6,0]
   + 2.044247600435395 betas[6,1] - 0.4186817893938694 betas[6,2]
   + 1.258880578516843 betas[6,3] - betas[6,4] + tau[13,6] = 0
 tau_computation_13_7: 0.7675764016808295 betas[7,0]
   + 2.044247600435395 betas[7,1] - 0.4186817893938694 betas[7,2]
   + 1.258880578516843 betas[7,3] - betas[7,4] + tau[13,7] = 0
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
 Define_Q_hat_13: - 1996.99414423995 R[13,0] - 2177.451927640506 R[13,1]
   - 2167.99485710778 R[13,2] - 2161.880787788705 R[13,3]
   - 2176.000122650908 R[13,4] - 2176.000217133179 R[13,5]
   - 2196.051792005164 R[13,6] - 2215.929809516887 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 120.316424069403 R[13,0] - 336.7812037801716 R[13,1]
   - 536.3200931222757 R[13,2] - 752.7848722574197 R[13,3]
   - 1049.958015646408 R[13,4] - 1266.422795036204 R[13,5]
   - 1493.711174243256 R[13,6] - 1782.999171908085 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 2095.612306618276 R[13,0] - 1879.147526907507 R[13,1]
   - 1679.608637565403 R[13,2] - 1463.143858430259 R[13,3]
   - 1165.970715041271 R[13,4] - 949.5059356514746 R[13,5]
   - 722.2175564444235 R[13,6] - 432.9295587795938 R[13,7] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: - Total_Demand_Up_to_K_minus_1[13] + Q0_var[13]
   + Lost_0_aux_13 = 0
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: - Total_Demand_from_k_to_T[13]
   + Q1_plus_left[13] + Lost_0_aux_13 = 0
 Profit_Constraint_13: - 700 Sold_0[13] + 700 Lost_0[13] - 700 Sold_1[13]
   + 300 Left_1[13] + 700 Lost_1[13] + profits_vars[13] = 0
 R486: - 0.3891679766292554 alphas[0] + 0.9663937193285366 alphas[1]
   + 0.0821648117046454 alphas[2] - 0.6729283363548018 alphas[3]
   - alphas[4] + f_var[14] = 0
 Q0_upper_bound_14:
   - 2238.523560606579 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 tau_computation_14_0: - 0.3891679766292554 betas[0,0]
   + 0.9663937193285366 betas[0,1] + 0.0821648117046454 betas[0,2]
   - 0.6729283363548018 betas[0,3] - betas[0,4] + tau[14,0] = 0
 tau_computation_14_1: - 0.3891679766292554 betas[1,0]
   + 0.9663937193285366 betas[1,1] + 0.0821648117046454 betas[1,2]
   - 0.6729283363548018 betas[1,3] - betas[1,4] + tau[14,1] = 0
 tau_computation_14_2: - 0.3891679766292554 betas[2,0]
   + 0.9663937193285366 betas[2,1] + 0.0821648117046454 betas[2,2]
   - 0.6729283363548018 betas[2,3] - betas[2,4] + tau[14,2] = 0
 tau_computation_14_3: - 0.3891679766292554 betas[3,0]
   + 0.9663937193285366 betas[3,1] + 0.0821648117046454 betas[3,2]
   - 0.6729283363548018 betas[3,3] - betas[3,4] + tau[14,3] = 0
 tau_computation_14_4: - 0.3891679766292554 betas[4,0]
   + 0.9663937193285366 betas[4,1] + 0.0821648117046454 betas[4,2]
   - 0.6729283363548018 betas[4,3] - betas[4,4] + tau[14,4] = 0
 tau_computation_14_5: - 0.3891679766292554 betas[5,0]
   + 0.9663937193285366 betas[5,1] + 0.0821648117046454 betas[5,2]
   - 0.6729283363548018 betas[5,3] - betas[5,4] + tau[14,5] = 0
 tau_computation_14_6: - 0.3891679766292554 betas[6,0]
   + 0.9663937193285366 betas[6,1] + 0.0821648117046454 betas[6,2]
   - 0.6729283363548018 betas[6,3] - betas[6,4] + tau[14,6] = 0
 tau_computation_14_7: - 0.3891679766292554 betas[7,0]
   + 0.9663937193285366 betas[7,1] + 0.0821648117046454 betas[7,2]
   - 0.6729283363548018 betas[7,3] - betas[7,4] + tau[14,7] = 0
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
 Define_Q_hat_14: - 2215.857376872711 R[14,0] - 1854.067676430988 R[14,1]
   - 1850.165394431478 R[14,2] - 1863.7684663524 R[14,3]
   - 1691.271836151716 R[14,4] - 1691.271949035237 R[14,5]
   - 1675.294508940131 R[14,6] - 1681.801050718364 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 156.5671715063123 R[14,0] - 335.8943276744028 R[14,1]
   - 546.2777514582446 R[14,2] - 725.6048806231415 R[14,3]
   - 835.9689685307987 R[14,4] - 1015.296097831371 R[14,5]
   - 1091.571429365784 R[14,6] - 1323.145713209427 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 1525.232800304258 R[14,0] - 1345.905644136168 R[14,1]
   - 1135.522220352326 R[14,2] - 956.1950911874293 R[14,3]
   - 845.8310032797721 R[14,4] - 666.5038739792 R[14,5]
   - 590.2285424447872 R[14,6] - 358.6542586011444 R[14,7] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: - Total_Demand_Up_to_K_minus_1[14] + Q0_var[14]
   + Lost_0_aux_14 = 0
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: - Total_Demand_from_k_to_T[14]
   + Q1_plus_left[14] + Lost_0_aux_14 = 0
 Profit_Constraint_14: - 700 Sold_0[14] + 700 Lost_0[14] - 700 Sold_1[14]
   + 300 Left_1[14] + 700 Lost_1[14] + profits_vars[14] = 0
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
 alphas[4] free
 betas[0,0] free
 betas[0,1] free
 betas[0,2] free
 betas[0,3] free
 betas[0,4] free
 betas[1,0] free
 betas[1,1] free
 betas[1,2] free
 betas[1,3] free
 betas[1,4] free
 betas[2,0] free
 betas[2,1] free
 betas[2,2] free
 betas[2,3] free
 betas[2,4] free
 betas[3,0] free
 betas[3,1] free
 betas[3,2] free
 betas[3,3] free
 betas[3,4] free
 betas[4,0] free
 betas[4,1] free
 betas[4,2] free
 betas[4,3] free
 betas[4,4] free
 betas[5,0] free
 betas[5,1] free
 betas[5,2] free
 betas[5,3] free
 betas[5,4] free
 betas[6,0] free
 betas[6,1] free
 betas[6,2] free
 betas[6,3] free
 betas[6,4] free
 betas[7,0] free
 betas[7,1] free
 betas[7,2] free
 betas[7,3] free
 betas[7,4] free
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
 Q0_var[0] <= 2239.523560606579
 Q0_var[1] <= 2239.523560606579
 Q0_var[2] <= 2239.523560606579
 Q0_var[3] <= 2239.523560606579
 Q0_var[4] <= 2239.523560606579
 Q0_var[5] <= 2239.523560606579
 Q0_var[6] <= 2239.523560606579
 Q0_var[7] <= 2239.523560606579
 Q0_var[8] <= 2239.523560606579
 Q0_var[9] <= 2239.523560606579
 Q0_var[10] <= 2239.523560606579
 Q0_var[11] <= 2239.523560606579
 Q0_var[12] <= 2239.523560606579
 Q0_var[13] <= 2239.523560606579
 Q0_var[14] <= 2239.523560606579
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
 logistic_constraint_0: Fraction_for_second_order_amount[0] =
   LOGISTIC ( f_var[0] )
 Max_Constraint_0: Q1_var[0] = MAX ( Q_hat_adjusted[0] , 0 )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Total_Demand_Up_to_K_minus_1[0] ,
   Q0_var[0] )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Total_Demand_from_k_to_T[0] ,
   Q1_plus_left[0] )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 logistic_constraint_1: Fraction_for_second_order_amount[1] =
   LOGISTIC ( f_var[1] )
 Max_Constraint_1: Q1_var[1] = MAX ( Q_hat_adjusted[1] , 0 )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Total_Demand_Up_to_K_minus_1[1] ,
   Q0_var[1] )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Total_Demand_from_k_to_T[1] ,
   Q1_plus_left[1] )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 logistic_constraint_2: Fraction_for_second_order_amount[2] =
   LOGISTIC ( f_var[2] )
 Max_Constraint_2: Q1_var[2] = MAX ( Q_hat_adjusted[2] , 0 )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Total_Demand_Up_to_K_minus_1[2] ,
   Q0_var[2] )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Total_Demand_from_k_to_T[2] ,
   Q1_plus_left[2] )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 logistic_constraint_3: Fraction_for_second_order_amount[3] =
   LOGISTIC ( f_var[3] )
 Max_Constraint_3: Q1_var[3] = MAX ( Q_hat_adjusted[3] , 0 )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Total_Demand_Up_to_K_minus_1[3] ,
   Q0_var[3] )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Total_Demand_from_k_to_T[3] ,
   Q1_plus_left[3] )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 logistic_constraint_4: Fraction_for_second_order_amount[4] =
   LOGISTIC ( f_var[4] )
 Max_Constraint_4: Q1_var[4] = MAX ( Q_hat_adjusted[4] , 0 )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Total_Demand_Up_to_K_minus_1[4] ,
   Q0_var[4] )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Total_Demand_from_k_to_T[4] ,
   Q1_plus_left[4] )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 logistic_constraint_5: Fraction_for_second_order_amount[5] =
   LOGISTIC ( f_var[5] )
 Max_Constraint_5: Q1_var[5] = MAX ( Q_hat_adjusted[5] , 0 )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Total_Demand_Up_to_K_minus_1[5] ,
   Q0_var[5] )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Total_Demand_from_k_to_T[5] ,
   Q1_plus_left[5] )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 logistic_constraint_6: Fraction_for_second_order_amount[6] =
   LOGISTIC ( f_var[6] )
 Max_Constraint_6: Q1_var[6] = MAX ( Q_hat_adjusted[6] , 0 )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Total_Demand_Up_to_K_minus_1[6] ,
   Q0_var[6] )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Total_Demand_from_k_to_T[6] ,
   Q1_plus_left[6] )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 logistic_constraint_7: Fraction_for_second_order_amount[7] =
   LOGISTIC ( f_var[7] )
 Max_Constraint_7: Q1_var[7] = MAX ( Q_hat_adjusted[7] , 0 )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Total_Demand_Up_to_K_minus_1[7] ,
   Q0_var[7] )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Total_Demand_from_k_to_T[7] ,
   Q1_plus_left[7] )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 logistic_constraint_8: Fraction_for_second_order_amount[8] =
   LOGISTIC ( f_var[8] )
 Max_Constraint_8: Q1_var[8] = MAX ( Q_hat_adjusted[8] , 0 )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Total_Demand_Up_to_K_minus_1[8] ,
   Q0_var[8] )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Total_Demand_from_k_to_T[8] ,
   Q1_plus_left[8] )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 logistic_constraint_9: Fraction_for_second_order_amount[9] =
   LOGISTIC ( f_var[9] )
 Max_Constraint_9: Q1_var[9] = MAX ( Q_hat_adjusted[9] , 0 )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Total_Demand_Up_to_K_minus_1[9] ,
   Q0_var[9] )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Total_Demand_from_k_to_T[9] ,
   Q1_plus_left[9] )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 logistic_constraint_10: Fraction_for_second_order_amount[10] =
   LOGISTIC ( f_var[10] )
 Max_Constraint_10: Q1_var[10] = MAX ( Q_hat_adjusted[10] , 0 )
 Constr_Sold_0_min_10: Sold_0[10] = MIN (
   Total_Demand_Up_to_K_minus_1[10] , Q0_var[10] )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Total_Demand_from_k_to_T[10] ,
   Q1_plus_left[10] )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 logistic_constraint_11: Fraction_for_second_order_amount[11] =
   LOGISTIC ( f_var[11] )
 Max_Constraint_11: Q1_var[11] = MAX ( Q_hat_adjusted[11] , 0 )
 Constr_Sold_0_min_11: Sold_0[11] = MIN (
   Total_Demand_Up_to_K_minus_1[11] , Q0_var[11] )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Total_Demand_from_k_to_T[11] ,
   Q1_plus_left[11] )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 logistic_constraint_12: Fraction_for_second_order_amount[12] =
   LOGISTIC ( f_var[12] )
 Max_Constraint_12: Q1_var[12] = MAX ( Q_hat_adjusted[12] , 0 )
 Constr_Sold_0_min_12: Sold_0[12] = MIN (
   Total_Demand_Up_to_K_minus_1[12] , Q0_var[12] )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Total_Demand_from_k_to_T[12] ,
   Q1_plus_left[12] )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 logistic_constraint_13: Fraction_for_second_order_amount[13] =
   LOGISTIC ( f_var[13] )
 Max_Constraint_13: Q1_var[13] = MAX ( Q_hat_adjusted[13] , 0 )
 Constr_Sold_0_min_13: Sold_0[13] = MIN (
   Total_Demand_Up_to_K_minus_1[13] , Q0_var[13] )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Total_Demand_from_k_to_T[13] ,
   Q1_plus_left[13] )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 logistic_constraint_14: Fraction_for_second_order_amount[14] =
   LOGISTIC ( f_var[14] )
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
