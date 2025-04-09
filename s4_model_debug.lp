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
   + profits_vars[14]
Subject To
 Fraction_lower_bound_0: Fraction_for_second_order_amount[0] >= 0
 Fraction_upper_bound_0: Fraction_for_second_order_amount[0] <= 1
 Q0_upper_bound_0: - 2238.523560606579 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 3048.717426668908 R[0,0] - 1829.265068551074 R[0,1]
   - 1732.442202225928 R[0,2] - 1771.01243279976 R[0,3]
   - 1637.899699940014 R[0,4] - 1637.89941138166 R[0,5]
   - 1556.827488078806 R[0,6] - 1516.535552040845 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 346.7940140898314 R[0,0] - 502.8161083731457 R[0,1]
   - 600.8119804692453 R[0,2] - 756.8342274363217 R[0,3]
   - 921.0513119249435 R[0,4] - 1077.073559033906 R[0,5]
   - 1204.491057178249 R[0,6] - 1204.491057178249 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1169.741537306342 R[0,0] - 1013.719443023028 R[0,1]
   - 915.7235709269286 R[0,2] - 759.7013239598522 R[0,3]
   - 595.4842394712304 R[0,4] - 439.461992362268 R[0,5]
   - 312.0444942179246 R[0,6] - 312.0444942179246 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 700 Sold_0[0] + 700 Lost_0[0] - 700 Sold_1[0]
   + 300 Left_1[0] + 700 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction_for_second_order_amount[1] >= 0
 Fraction_upper_bound_1: Fraction_for_second_order_amount[1] <= 1
 Q0_upper_bound_1: - 2238.523560606579 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 2575.487540101392 R[1,0] - 2073.757778177428 R[1,1]
   - 2047.381575421603 R[1,2] - 2015.465938143341 R[1,3]
   - 2002.071075363615 R[1,4] - 2002.071287314232 R[1,5]
   - 2099.730874649836 R[1,6] - 2011.75992831364 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 242.453034319292 R[1,0] - 431.124975109031 R[1,1]
   - 612.284342155843 R[1,2] - 800.956294192751 R[1,3]
   - 1049.332968991489 R[1,4] - 1238.004921156844 R[1,5]
   - 1567.080808843288 R[1,6] - 1634.416023523455 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 1769.306893534873 R[1,0] - 1580.634952745134 R[1,1]
   - 1399.475585698322 R[1,2] - 1210.803633661414 R[1,3]
   - 962.4269588626755 R[1,4] - 773.7550066973211 R[1,5]
   - 444.6791190108769 R[1,6] - 377.3439043307095 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 700 Sold_0[1] + 700 Lost_0[1] - 700 Sold_1[1]
   + 300 Left_1[1] + 700 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction_for_second_order_amount[2] >= 0
 Fraction_upper_bound_2: Fraction_for_second_order_amount[2] <= 1
 Q0_upper_bound_2: - 2238.523560606579 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 1780.498863909008 R[2,0] - 1363.856179026433 R[2,1]
   - 1333.584982228291 R[2,2] - 1236.727742756625 R[2,3]
   - 1298.041891371108 R[2,4] - 1298.042392464578 R[2,5]
   - 1280.811083702263 R[2,6] - 1216.479406663358 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 67.16846389188552 R[2,0] - 188.9193807969859 R[2,1]
   - 264.1640153908049 R[2,2] - 385.9268949143069 R[2,3]
   - 653.1715430539091 R[2,4] - 774.9344230338165 R[2,5]
   - 911.7303900052442 R[2,6] - 972.9536465364326 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1149.310942604362 R[2,0] - 1027.560025699262 R[2,1]
   - 952.3153911054426 R[2,2] - 830.5525115819405 R[2,3]
   - 563.3078634423384 R[2,4] - 441.544983462431 R[2,5]
   - 304.7490164910033 R[2,6] - 243.5257599598149 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 700 Sold_0[2] + 700 Lost_0[2] - 700 Sold_1[2]
   + 300 Left_1[2] + 700 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction_for_second_order_amount[3] >= 0
 Fraction_upper_bound_3: Fraction_for_second_order_amount[3] <= 1
 Q0_upper_bound_3: - 2238.523560606579 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 2122.775617657253 R[3,0] - 2369.653931170435 R[3,1]
   - 2293.973774048757 R[3,2] - 2272.171129380989 R[3,3]
   - 2355.408920023375 R[3,4] - 2355.409125105393 R[3,5]
   - 2258.958297781131 R[3,6] - 2157.877328482206 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 142.6359963797691 R[3,0] - 369.2796675129249 R[3,1]
   - 487.7666325364999 R[3,2] - 714.4103088833405 R[3,3]
   - 1089.549806606284 R[3,4] - 1316.193483174528 R[3,5]
   - 1529.271048219172 R[3,6] - 1704.589974475013 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 2015.241331231734 R[3,0] - 1788.597660098578 R[3,1]
   - 1670.110695075003 R[3,2] - 1443.467018728162 R[3,3]
   - 1068.327521005219 R[3,4] - 841.6838444369743 R[3,5]
   - 628.6062793923305 R[3,6] - 453.2873531364895 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 700 Sold_0[3] + 700 Lost_0[3] - 700 Sold_1[3]
   + 300 Left_1[3] + 700 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction_for_second_order_amount[4] >= 0
 Fraction_upper_bound_4: Fraction_for_second_order_amount[4] <= 1
 Q0_upper_bound_4: - 2238.523560606579 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 2012.657477104841 R[4,0] - 1685.471894958457 R[4,1]
   - 1681.329662082542 R[4,2] - 1559.108107091762 R[4,3]
   - 1600.565472683536 R[4,4] - 1600.565961557245 R[4,5]
   - 1560.922023531124 R[4,6] - 1583.58923869453 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 118.3563914733927 R[4,0] - 272.0707178282555 R[4,1]
   - 419.104491262575 R[4,2] - 572.8187086504394 R[4,3]
   - 829.4105490628526 R[4,4] - 983.1247663962437 R[4,5]
   - 1105.190125377105 R[4,6] - 1276.160803230898 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 1465.232846424287 R[4,0] - 1311.518520069425 R[4,1]
   - 1164.484746635105 R[4,2] - 1010.770529247241 R[4,3]
   - 754.1786888348274 R[4,4] - 600.4644715014364 R[4,5]
   - 478.399112520575 R[4,6] - 307.4284346667821 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 700 Sold_0[4] + 700 Lost_0[4] - 700 Sold_1[4]
   + 300 Left_1[4] + 700 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction_for_second_order_amount[5] >= 0
 Fraction_upper_bound_5: Fraction_for_second_order_amount[5] <= 1
 Q0_upper_bound_5: - 2238.523560606579 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] = 1
 Define_Q_hat_5: - 2825.63856775645 R[5,0] - 1656.685515097987 R[5,1]
   - 1672.641194812733 R[5,2] - 1607.992316971436 R[5,3]
   - 1692.845856895295 R[5,4] - 1692.84604542739 R[5,5]
   - 1697.016560877556 R[5,6] - 1686.865789111867 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 297.6080531737747 R[5,0] - 437.7446355089113 R[5,1]
   - 651.9091539350222 R[5,2] - 792.0455438941495 R[5,3]
   - 1026.563765506844 R[5,4] - 1166.700155628317 R[5,5]
   - 1359.833183857264 R[5,6] - 1406.593008140711 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 1389.257735209883 R[5,0] - 1249.121152874746 R[5,1]
   - 1034.956634448635 R[5,2] - 894.8202444895078 R[5,3]
   - 660.3020228768136 R[5,4] - 520.1656327553405 R[5,5]
   - 327.0326045263934 R[5,6] - 280.2727802429461 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 700 Sold_0[5] + 700 Lost_0[5] - 700 Sold_1[5]
   + 300 Left_1[5] + 700 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction_for_second_order_amount[6] >= 0
 Fraction_upper_bound_6: Fraction_for_second_order_amount[6] <= 1
 Q0_upper_bound_6: - 2238.523560606579 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] = 1
 Define_Q_hat_6: - 2467.499819775342 R[6,0] - 2691.306673436467 R[6,1]
   - 2751.262242304367 R[6,2] - 2696.802187285706 R[6,3]
   - 2619.107580790297 R[6,4] - 2619.107101709872 R[6,5]
   - 2629.529956948356 R[6,6] - 2692.57068970506 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 218.6431591152256 R[6,0] - 475.8073876034488 R[6,1]
   - 810.404288177731 R[6,2] - 1067.568517590319 R[6,3]
   - 1307.545386653039 R[6,4] - 1564.709615291974 R[6,5]
   - 1778.9135570918 R[6,6] - 2178.242231608116 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 2473.927529770759 R[6,0] - 2216.763301282535 R[6,1]
   - 1882.166400708254 R[6,2] - 1625.002171295666 R[6,3]
   - 1385.025302232945 R[6,4] - 1127.861073594011 R[6,5]
   - 913.6571317941842 R[6,6] - 514.3284572778689 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 700 Sold_0[6] + 700 Lost_0[6] - 700 Sold_1[6]
   + 300 Left_1[6] + 700 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction_for_second_order_amount[7] >= 0
 Fraction_upper_bound_7: Fraction_for_second_order_amount[7] <= 1
 Q0_upper_bound_7: - 2238.523560606579 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] = 1
 Define_Q_hat_7: - 2516.156659370979 R[7,0] - 2918.643351259354 R[7,1]
   - 2972.397351636786 R[7,2] - 2941.243901003038 R[7,3]
   - 2809.939303806531 R[7,4] - 2809.940063392454 R[7,5]
   - 2732.343076616552 R[7,6] - 2658.215284916394 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 229.3713537201166 R[7,0] - 510.6130812653693 R[7,1]
   - 858.5530690051587 R[7,2] - 1139.794796894254 R[7,3]
   - 1371.800107890618 R[7,4] - 1653.041835878916 R[7,5]
   - 1783.687745166626 R[7,6] - 2095.731828745784 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 2428.843931002263 R[7,0] - 2147.60220345701 R[7,1]
   - 1799.662215717221 R[7,2] - 1518.420487828126 R[7,3]
   - 1286.415176831762 R[7,4] - 1005.173448843464 R[7,5]
   - 874.5275395557533 R[7,6] - 562.483455976596 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 700 Sold_0[7] + 700 Lost_0[7] - 700 Sold_1[7]
   + 300 Left_1[7] + 700 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction_for_second_order_amount[8] >= 0
 Fraction_upper_bound_8: Fraction_for_second_order_amount[8] <= 1
 Q0_upper_bound_8: - 2238.523560606579 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] = 1
 Define_Q_hat_8: - 1544.294578573674 R[8,0] - 780.3366994738826 R[8,1]
   - 855.2081852771237 R[8,2] - 1095.900295524044 R[8,3]
   - 1081.128010489874 R[8,4] - 1081.127704289419 R[8,5]
   - 1073.818002857428 R[8,6] - 1044.672592713792 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 15.08851869516009 R[8,0] - 76.46897460819714 R[8,1]
   - 224.0371926348129 R[8,2] - 285.5390199720595 R[8,3]
   - 503.1240500938345 R[8,4] - 564.6258814917302 R[8,5]
   - 602.1562130341003 R[8,6] - 921.6689358427677 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 1029.584072591201 R[8,0] - 968.2036166781641 R[8,1]
   - 820.6353986515484 R[8,2] - 759.1335713143018 R[8,3]
   - 541.5485411925267 R[8,4] - 480.046709794631 R[8,5]
   - 442.5163782522608 R[8,6] - 123.0036554435935 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 700 Sold_0[8] + 700 Lost_0[8] - 700 Sold_1[8]
   + 300 Left_1[8] + 700 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction_for_second_order_amount[9] >= 0
 Fraction_upper_bound_9: Fraction_for_second_order_amount[9] <= 1
 Q0_upper_bound_9: - 2238.523560606579 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] = 1
 Define_Q_hat_9: - 1681.410246319555 R[9,0] - 1226.875901019048 R[9,1]
   - 1230.516018699582 R[9,2] - 1114.901619176834 R[9,3]
   - 1028.300436999323 R[9,4] - 1028.29983985408 R[9,5]
   - 975.5154829839582 R[9,6] - 991.8313852288736 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 45.32072405790363 R[9,0] - 153.4607028816868 R[9,1]
   - 254.96369282744 R[9,2] - 363.1204415307662 R[9,3]
   - 524.3385785588121 R[9,4] - 632.4953267662534 R[9,5]
   - 632.4953267662534 R[9,6] - 775.5178881293125 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 946.5106604862918 R[9,0] - 838.3706816625088 R[9,1]
   - 736.8676917167556 R[9,2] - 628.7109430134293 R[9,3]
   - 467.4928059853835 R[9,4] - 359.3360577779422 R[9,5]
   - 359.3360577779422 R[9,6] - 216.313496414883 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 700 Sold_0[9] + 700 Lost_0[9] - 700 Sold_1[9]
   + 300 Left_1[9] + 700 Lost_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction_for_second_order_amount[10] >= 0
 Fraction_upper_bound_10: Fraction_for_second_order_amount[10] <= 1
 Q0_upper_bound_10:
   - 2238.523560606579 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 Ensure_only_one_R_true_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4]
   + R[10,5] + R[10,6] + R[10,7] = 1
 Define_Q_hat_10: - 2091.8085916863 R[10,0] - 1542.210781341673 R[10,1]
   - 1503.518234752883 R[10,2] - 1400.735977080048 R[10,3]
   - 1301.570061425497 R[10,4] - 1301.570894020447 R[10,5]
   - 1330.72173902345 R[10,6] - 1299.500463027703 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 135.8081735348821 R[10,0] - 272.9380460281882 R[10,1]
   - 368.9828224614398 R[10,2] - 506.1126646646167 R[10,3]
   - 678.2290734058711 R[10,4] - 815.3589157790119 R[10,5]
   - 969.1004044584502 R[10,6] - 1025.240777592752 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 1163.692288804152 R[10,0] - 1026.562416310846 R[10,1]
   - 930.5176398775944 R[10,2] - 793.3877976744175 R[10,3]
   - 621.2713889331632 R[10,4] - 484.1415465600223 R[10,5]
   - 330.4000578805841 R[10,6] - 274.2596847462816 R[10,7] = 0
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
 Fraction_lower_bound_11: Fraction_for_second_order_amount[11] >= 0
 Fraction_upper_bound_11: Fraction_for_second_order_amount[11] <= 1
 Q0_upper_bound_11:
   - 2238.523560606579 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 Ensure_only_one_R_true_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4]
   + R[11,5] + R[11,6] + R[11,7] = 1
 Define_Q_hat_11: - 2285.518074260928 R[11,0] - 2638.282441377046 R[11,1]
   - 2510.957451652625 R[11,2] - 2553.720150178149 R[11,3]
   - 2529.335128596438 R[11,4] - 2529.334791773525 R[11,5]
   - 2610.90314006978 R[11,6] - 2617.13070339671 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 178.5185723841257 R[11,0] - 432.260192795386 R[11,1]
   - 516.291430861939 R[11,2] - 770.0330506835021 R[11,3]
   - 1124.531996970793 R[11,4] - 1378.273616813355 R[11,5]
   - 1817.556415571177 R[11,6] - 2109.647463356403 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 2438.612130657401 R[11,0] - 2184.870510246141 R[11,1]
   - 2100.839272179588 R[11,2] - 1847.097652358025 R[11,3]
   - 1492.598706070734 R[11,4] - 1238.857086228172 R[11,5]
   - 799.5742874703494 R[11,6] - 507.4832396851237 R[11,7] = 0
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
 Fraction_lower_bound_12: Fraction_for_second_order_amount[12] >= 0
 Fraction_upper_bound_12: Fraction_for_second_order_amount[12] <= 1
 Q0_upper_bound_12:
   - 2238.523560606579 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 Ensure_only_one_R_true_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4]
   + R[12,5] + R[12,6] + R[12,7] = 1
 Define_Q_hat_12: - 2395.016255185027 R[12,0] - 3043.562873274061 R[12,1]
   - 2995.914880924804 R[12,2] - 3025.597180873808 R[12,3]
   - 2979.234647562913 R[12,4] - 2979.23450072785 R[12,5]
   - 2975.083420818698 R[12,6] - 2926.002364698358 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 202.6614843282177 R[12,0] - 499.0343414942936 R[12,1]
   - 722.6025865509852 R[12,2] - 1018.975443543831 R[12,3]
   - 1361.86448443228 R[12,4] - 1658.237341343104 R[12,5]
   - 1987.550861929503 R[12,6] - 2333.256650697667 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 2723.340880191098 R[12,0] - 2426.968023025022 R[12,1]
   - 2203.399777968331 R[12,2] - 1907.026920975484 R[12,3]
   - 1564.137880087036 R[12,4] - 1267.765023176211 R[12,5]
   - 938.4515025898131 R[12,6] - 592.7457138216488 R[12,7] = 0
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
 Fraction_lower_bound_13: Fraction_for_second_order_amount[13] >= 0
 Fraction_upper_bound_13: Fraction_for_second_order_amount[13] <= 1
 Q0_upper_bound_13:
   - 2238.523560606579 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 Ensure_only_one_R_true_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4]
   + R[13,5] + R[13,6] + R[13,7] = 1
 Define_Q_hat_13: - 2955.142408023167 R[13,0] - 3071.812963837329 R[13,1]
   - 3045.862990246404 R[13,2] - 3113.74505900033 R[13,3]
   - 3199.571498034994 R[13,4] - 3199.571593128333 R[13,5]
   - 3137.229913713267 R[13,6] - 3192.54567566019 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 326.1619504677251 R[13,0] - 618.4048766384957 R[13,1]
   - 902.052844729998 R[13,2] - 1194.295770404813 R[13,3]
   - 1582.859383582063 R[13,4] - 1875.102309749651 R[13,5]
   - 2196.896637584514 R[13,6] - 2608.059823035458 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 2866.383724902909 R[13,0] - 2574.140798732138 R[13,1]
   - 2290.492830640636 R[13,2] - 1998.249904965821 R[13,3]
   - 1609.686291788571 R[13,4] - 1317.443365620984 R[13,5]
   - 995.6490377861202 R[13,6] - 584.4858523351759 R[13,7] = 0
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
 Fraction_lower_bound_14: Fraction_for_second_order_amount[14] >= 0
 Fraction_upper_bound_14: Fraction_for_second_order_amount[14] <= 1
 Q0_upper_bound_14:
   - 2238.523560606579 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 Ensure_only_one_R_true_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4]
   + R[14,5] + R[14,6] + R[14,7] = 1
 Define_Q_hat_14: - 2904.431376197994 R[14,0] - 2918.515382012934 R[14,1]
   - 3057.337517350244 R[14,2] - 3007.62563263891 R[14,3]
   - 3022.656958577538 R[14,4] - 3022.656928462895 R[14,5]
   - 3039.421594261383 R[14,6] - 2959.028733168197 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 314.9808334334807 R[14,0] - 591.2179193534751 R[14,1]
   - 1067.580495655978 R[14,2] - 1343.817582421547 R[14,3]
   - 1607.081185210258 R[14,4] - 1883.31827159774 R[14,5]
   - 2121.692046842761 R[14,6] - 2406.554560121022 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 2644.047899462505 R[14,0] - 2367.810813542511 R[14,1]
   - 1891.448237240008 R[14,2] - 1615.211150474439 R[14,3]
   - 1351.947547685728 R[14,4] - 1075.710461298246 R[14,5]
   - 837.3366860532249 R[14,6] - 552.4741727749634 R[14,7] = 0
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
Bounds
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
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
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
 Max_Constraint_0: Q1_var[0] = MAX ( Q_hat_adjusted[0] , 0 )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Total_Demand_Up_to_K_minus_1[0] ,
   Q0_var[0] )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Total_Demand_from_k_to_T[0] ,
   Q1_plus_left[0] )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 Max_Constraint_1: Q1_var[1] = MAX ( Q_hat_adjusted[1] , 0 )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Total_Demand_Up_to_K_minus_1[1] ,
   Q0_var[1] )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Total_Demand_from_k_to_T[1] ,
   Q1_plus_left[1] )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 Max_Constraint_2: Q1_var[2] = MAX ( Q_hat_adjusted[2] , 0 )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Total_Demand_Up_to_K_minus_1[2] ,
   Q0_var[2] )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Total_Demand_from_k_to_T[2] ,
   Q1_plus_left[2] )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 Max_Constraint_3: Q1_var[3] = MAX ( Q_hat_adjusted[3] , 0 )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Total_Demand_Up_to_K_minus_1[3] ,
   Q0_var[3] )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Total_Demand_from_k_to_T[3] ,
   Q1_plus_left[3] )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 Max_Constraint_4: Q1_var[4] = MAX ( Q_hat_adjusted[4] , 0 )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Total_Demand_Up_to_K_minus_1[4] ,
   Q0_var[4] )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Total_Demand_from_k_to_T[4] ,
   Q1_plus_left[4] )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 Max_Constraint_5: Q1_var[5] = MAX ( Q_hat_adjusted[5] , 0 )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Total_Demand_Up_to_K_minus_1[5] ,
   Q0_var[5] )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Total_Demand_from_k_to_T[5] ,
   Q1_plus_left[5] )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 Max_Constraint_6: Q1_var[6] = MAX ( Q_hat_adjusted[6] , 0 )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Total_Demand_Up_to_K_minus_1[6] ,
   Q0_var[6] )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Total_Demand_from_k_to_T[6] ,
   Q1_plus_left[6] )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 Max_Constraint_7: Q1_var[7] = MAX ( Q_hat_adjusted[7] , 0 )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Total_Demand_Up_to_K_minus_1[7] ,
   Q0_var[7] )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Total_Demand_from_k_to_T[7] ,
   Q1_plus_left[7] )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 Max_Constraint_8: Q1_var[8] = MAX ( Q_hat_adjusted[8] , 0 )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Total_Demand_Up_to_K_minus_1[8] ,
   Q0_var[8] )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Total_Demand_from_k_to_T[8] ,
   Q1_plus_left[8] )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 Max_Constraint_9: Q1_var[9] = MAX ( Q_hat_adjusted[9] , 0 )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Total_Demand_Up_to_K_minus_1[9] ,
   Q0_var[9] )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Total_Demand_from_k_to_T[9] ,
   Q1_plus_left[9] )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 Max_Constraint_10: Q1_var[10] = MAX ( Q_hat_adjusted[10] , 0 )
 Constr_Sold_0_min_10: Sold_0[10] = MIN (
   Total_Demand_Up_to_K_minus_1[10] , Q0_var[10] )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Total_Demand_from_k_to_T[10] ,
   Q1_plus_left[10] )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 Max_Constraint_11: Q1_var[11] = MAX ( Q_hat_adjusted[11] , 0 )
 Constr_Sold_0_min_11: Sold_0[11] = MIN (
   Total_Demand_Up_to_K_minus_1[11] , Q0_var[11] )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Total_Demand_from_k_to_T[11] ,
   Q1_plus_left[11] )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 Max_Constraint_12: Q1_var[12] = MAX ( Q_hat_adjusted[12] , 0 )
 Constr_Sold_0_min_12: Sold_0[12] = MIN (
   Total_Demand_Up_to_K_minus_1[12] , Q0_var[12] )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Total_Demand_from_k_to_T[12] ,
   Q1_plus_left[12] )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 Max_Constraint_13: Q1_var[13] = MAX ( Q_hat_adjusted[13] , 0 )
 Constr_Sold_0_min_13: Sold_0[13] = MIN (
   Total_Demand_Up_to_K_minus_1[13] , Q0_var[13] )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Total_Demand_from_k_to_T[13] ,
   Q1_plus_left[13] )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
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
