\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 Holding_Cost_0[0] + 0 Holding_Cost_0[1] + 0 Holding_Cost_0[2]
   + 0 Holding_Cost_0[3] + 0 Holding_Cost_0[4] + 0 Holding_Cost_0[5]
   + 0 Holding_Cost_0[6] + 0 Holding_Cost_0[7] + 0 Holding_Cost_0[8]
   + 0 Holding_Cost_0[9] + 0 Holding_Cost_0[10] + 0 Holding_Cost_0[11]
   + 0 Holding_Cost_0[12] + 0 Holding_Cost_0[13] + 0 Holding_Cost_0[14]
   + 0 Holding_Cost_0[15] + 0 Holding_Cost_0[16] + 0 Holding_Cost_0[17]
   + 0 Holding_Cost_0[18] + 0 Holding_Cost_0[19] + 0 Holding_Cost_0[20]
   + 0 Holding_Cost_0[21] + 0 Holding_Cost_0[22] + 0 Holding_Cost_0[23]
   + 0 Holding_Cost_0[24] + 0 Holding_Cost_0[25] + 0 Holding_Cost_0[26]
   + 0 Holding_Cost_0[27] + 0 Holding_Cost_0[28] + 0 Holding_Cost_0[29]
   + 0 Holding_Cost_1[0] + 0 Holding_Cost_1[1] + 0 Holding_Cost_1[2]
   + 0 Holding_Cost_1[3] + 0 Holding_Cost_1[4] + 0 Holding_Cost_1[5]
   + 0 Holding_Cost_1[6] + 0 Holding_Cost_1[7] + 0 Holding_Cost_1[8]
   + 0 Holding_Cost_1[9] + 0 Holding_Cost_1[10] + 0 Holding_Cost_1[11]
   + 0 Holding_Cost_1[12] + 0 Holding_Cost_1[13] + 0 Holding_Cost_1[14]
   + 0 Holding_Cost_1[15] + 0 Holding_Cost_1[16] + 0 Holding_Cost_1[17]
   + 0 Holding_Cost_1[18] + 0 Holding_Cost_1[19] + 0 Holding_Cost_1[20]
   + 0 Holding_Cost_1[21] + 0 Holding_Cost_1[22] + 0 Holding_Cost_1[23]
   + 0 Holding_Cost_1[24] + 0 Holding_Cost_1[25] + 0 Holding_Cost_1[26]
   + 0 Holding_Cost_1[27] + 0 Holding_Cost_1[28] + 0 Holding_Cost_1[29]
   + profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14] + profits_vars[15] + profits_vars[16]
   + profits_vars[17] + profits_vars[18] + profits_vars[19]
   + profits_vars[20] + profits_vars[21] + profits_vars[22]
   + profits_vars[23] + profits_vars[24] + profits_vars[25]
   + profits_vars[26] + profits_vars[27] + profits_vars[28]
   + profits_vars[29]
Subject To
 Fraction_lower_bound_0: Fraction_for_second_order_amount[0] >= 0
 Fraction_upper_bound_0: Fraction_for_second_order_amount[0] <= 1
 Q0_upper_bound_0: - 2834.593301140667 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] + R[0,8] = 1
 Define_Q_hat_0: - 624.2789400376344 R[0,0] - 624.6933737557733 R[0,1]
   - 623.7507120589721 R[0,2] - 620.8390770055479 R[0,3]
   - 617.7342331885158 R[0,4] - 617.7342331952448 R[0,5]
   - 618.4554379348677 R[0,6] - 616.341580645932 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 60.05506882576569 R[0,0] - 121.8983290333844 R[0,1]
   - 184.3678164972877 R[0,2] - 246.179326105148 R[0,3]
   - 306.9144447287598 R[0,4] - 368.7259543449988 R[0,5]
   - 430.8494372252158 R[0,6] - 492.7185608408271 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 556.2865117241895 R[0,0] - 494.4432515165709 R[0,1]
   - 431.9737640526674 R[0,2] - 370.1622544448072 R[0,3]
   - 309.4271358211953 R[0,4] - 247.6156262049564 R[0,5]
   - 185.4921433247394 R[0,6] - 123.6230197091281 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 600 Sold_0[0] + 600 Lost_0[0] - 600 Sold_1[0]
   + 400 Left_1[0] + 600 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction_for_second_order_amount[1] >= 0
 Fraction_upper_bound_1: Fraction_for_second_order_amount[1] <= 1
 Q0_upper_bound_1: - 2834.593301140667 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] + R[1,8] = 1
 Define_Q_hat_1: - 687.7509310672793 R[1,0] - 684.9042835447847 R[1,1]
   - 686.2733080905375 R[1,2] - 683.3941152754651 R[1,3]
   - 681.8018731757063 R[1,4] - 681.801873175325 R[1,5]
   - 681.8524953763392 R[1,6] - 682.9018161718318 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 73.12377837899443 R[1,0] - 141.4461530178674 R[1,1]
   - 209.8099083416733 R[1,2] - 277.6953634657769 R[1,3]
   - 341.1080347964092 R[1,4] - 408.9934899194309 R[1,5]
   - 479.9420806299379 R[1,6] - 547.1309081247325 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 609.778039424712 R[1,0] - 541.455664785839 R[1,1]
   - 473.0919094620332 R[1,2] - 405.2064543379296 R[1,3]
   - 341.7937830072972 R[1,4] - 273.9083278842756 R[1,5]
   - 202.9597371737685 R[1,6] - 135.770909678974 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 600 Sold_0[1] + 600 Lost_0[1] - 600 Sold_1[1]
   + 400 Left_1[1] + 600 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction_for_second_order_amount[2] >= 0
 Fraction_upper_bound_2: Fraction_for_second_order_amount[2] <= 1
 Q0_upper_bound_2: - 2834.593301140667 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] + R[2,8] = 1
 Define_Q_hat_2: - 617.1659996705411 R[2,0] - 595.7740397476126 R[2,1]
   - 595.1001454330392 R[2,2] - 605.9629080297345 R[2,3]
   - 605.9850078490618 R[2,4] - 605.9850078618884 R[2,5]
   - 606.5180824114365 R[2,6] - 607.2671281949398 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 58.59053383609501 R[2,0] - 116.2499573535704 R[2,1]
   - 174.8766195964306 R[2,2] - 235.4451003518676 R[2,3]
   - 302.4298521682813 R[2,4] - 362.9983329362325 R[2,5]
   - 424.4776754986781 R[2,6] - 486.1301728922332 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 548.6765901689786 R[2,0] - 491.0171666515031 R[2,1]
   - 432.3905044086429 R[2,2] - 371.8220236532059 R[2,3]
   - 304.8372718367922 R[2,4] - 244.268791068841 R[2,5]
   - 182.7894485063954 R[2,6] - 121.1369511128403 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 600 Sold_0[2] + 600 Lost_0[2] - 600 Sold_1[2]
   + 400 Left_1[2] + 600 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction_for_second_order_amount[3] >= 0
 Fraction_upper_bound_3: Fraction_for_second_order_amount[3] <= 1
 Q0_upper_bound_3: - 2834.593301140667 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] + R[3,8] = 1
 Define_Q_hat_3: - 687.5889845403087 R[3,0] - 683.0060278794697 R[3,1]
   - 681.523235171587 R[3,2] - 676.0555414390212 R[3,3]
   - 677.6239060678356 R[3,4] - 677.6239060746518 R[3,5]
   - 678.1961261812642 R[3,6] - 675.9732079654088 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 73.09043403151453 R[3,0] - 141.124197678327 R[3,1]
   - 212.257656270999 R[3,2] - 279.2276286846237 R[3,3]
   - 343.4084257127438 R[3,4] - 410.3783981302215 R[3,5]
   - 477.5923632567367 R[3,6] - 542.03326594092 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 602.882772021925 R[3,0] - 534.8490083751126 R[3,1]
   - 463.7155497824406 R[3,2] - 396.7455773688158 R[3,3]
   - 332.5647803406957 R[3,4] - 265.5948079232181 R[3,5]
   - 198.3808427967028 R[3,6] - 133.9399401125195 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 600 Sold_0[3] + 600 Lost_0[3] - 600 Sold_1[3]
   + 400 Left_1[3] + 600 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction_for_second_order_amount[4] >= 0
 Fraction_upper_bound_4: Fraction_for_second_order_amount[4] <= 1
 Q0_upper_bound_4: - 2834.593301140667 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] + R[4,8] = 1
 Define_Q_hat_4: - 638.5142752737482 R[4,0] - 663.1503907502852 R[4,1]
   - 664.558353792233 R[4,2] - 668.7783034959282 R[4,3]
   - 668.7567324661085 R[4,4] - 668.7567324698065 R[4,5]
   - 669.6162582625476 R[4,6] - 670.9129379522215 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 62.98608546122504 R[4,0] - 130.1739640369019 R[4,1]
   - 194.9965743890988 R[4,2] - 262.783528348855 R[4,3]
   - 328.7755036076944 R[4,4] - 396.5624575668369 R[4,5]
   - 466.0990519795713 R[4,6] - 535.3390257094343 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 607.9268496893519 R[4,0] - 540.738971113675 R[4,1]
   - 475.9163607614781 R[4,2] - 408.1294068017219 R[4,3]
   - 342.1374315428824 R[4,4] - 274.35047758374 R[4,5]
   - 204.8138831710056 R[4,6] - 135.5739094411426 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 600 Sold_0[4] + 600 Lost_0[4] - 600 Sold_1[4]
   + 400 Left_1[4] + 600 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction_for_second_order_amount[5] >= 0
 Fraction_upper_bound_5: Fraction_for_second_order_amount[5] <= 1
 Q0_upper_bound_5: - 2834.593301140667 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] + R[5,8] = 1
 Define_Q_hat_5: - 655.9425432466721 R[5,0] - 665.5054732258357 R[5,1]
   - 665.7409350331512 R[5,2] - 671.9443105808322 R[5,3]
   - 675.5951733755874 R[5,4] - 675.5951733787201 R[5,5]
   - 673.381777933745 R[5,6] - 673.8424927228542 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 66.57451822438826 R[5,0] - 133.3303861542171 R[5,1]
   - 199.9910180652321 R[5,2] - 267.7650936858963 R[5,3]
   - 336.8494288287964 R[5,4] - 404.6235044474996 R[5,5]
   - 471.4971553371083 R[5,6] - 538.294341710985 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 607.2679737721083 R[5,0] - 540.5121058422795 R[5,1]
   - 473.8514739312645 R[5,2] - 406.0773983106004 R[5,3]
   - 336.9930631677003 R[5,4] - 269.2189875489971 R[5,5]
   - 202.3453366593884 R[5,6] - 135.5481502855116 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 600 Sold_0[5] + 600 Lost_0[5] - 600 Sold_1[5]
   + 400 Left_1[5] + 600 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction_for_second_order_amount[6] >= 0
 Fraction_upper_bound_6: Fraction_for_second_order_amount[6] <= 1
 Q0_upper_bound_6: - 2834.593301140667 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] + R[6,8] = 1
 Define_Q_hat_6: - 628.576091668707 R[6,0] - 625.1930038597293 R[6,1]
   - 626.2860802402768 R[6,2] - 627.4648028717293 R[6,3]
   - 634.5676801438115 R[6,4] - 634.5676801473867 R[6,5]
   - 633.2392638398837 R[6,6] - 634.3015606352475 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 60.9398406071715 R[6,0] - 122.6639442744694 R[6,1]
   - 183.1332656235871 R[6,2] - 245.7274912707341 R[6,3]
   - 314.6681714189907 R[6,4] - 377.2623970634454 R[6,5]
   - 442.9465741026174 R[6,6] - 509.1131115475491 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 573.3617171100173 R[6,0] - 511.6376134427194 R[6,1]
   - 451.1682920936017 R[6,2] - 388.5740664464547 R[6,3]
   - 319.633386298198 R[6,4] - 257.0391606537434 R[6,5]
   - 191.3549836145713 R[6,6] - 125.1884461696397 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 600 Sold_0[6] + 600 Lost_0[6] - 600 Sold_1[6]
   + 400 Left_1[6] + 600 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction_for_second_order_amount[7] >= 0
 Fraction_upper_bound_7: Fraction_for_second_order_amount[7] <= 1
 Q0_upper_bound_7: - 2834.593301140667 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] + R[7,8] = 1
 Define_Q_hat_7: - 668.9469683562171 R[7,0] - 651.4913898979582 R[7,1]
   - 652.0700124964735 R[7,2] - 647.3966293341236 R[7,3]
   - 641.6766401956468 R[7,4] - 641.6766401993377 R[7,5]
   - 638.7621882450903 R[7,6] - 643.1075373536681 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 69.25209382752287 R[7,0] - 133.226020508261 R[7,1]
   - 198.1087985104492 R[7,2] - 261.8469298809117 R[7,3]
   - 321.3280100786132 R[7,4] - 385.0661414565567 R[7,5]
   - 449.3507697924065 R[7,6] - 515.6312800801095 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 573.8554430216222 R[7,0] - 509.8815163408841 R[7,1]
   - 444.998738338696 R[7,2] - 381.2606069682334 R[7,3]
   - 321.7795267705318 R[7,4] - 258.0413953925884 R[7,5]
   - 193.7567670567385 R[7,6] - 127.4762567690356 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 600 Sold_0[7] + 600 Lost_0[7] - 600 Sold_1[7]
   + 400 Left_1[7] + 600 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction_for_second_order_amount[8] >= 0
 Fraction_upper_bound_8: Fraction_for_second_order_amount[8] <= 1
 Q0_upper_bound_8: - 2834.593301140667 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] + R[8,8] = 1
 Define_Q_hat_8: - 635.0858931980144 R[8,0] - 652.7396395201881 R[8,1]
   - 654.5200260586812 R[8,2] - 654.6237770707921 R[8,3]
   - 650.3970663596465 R[8,4] - 650.3970663532273 R[8,5]
   - 653.4102655760998 R[8,6] - 651.9164231345428 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 62.28019095786315 R[8,0] - 128.0017163072239 R[8,1]
   - 191.1349128248177 R[8,2] - 257.0109091777701 R[8,3]
   - 318.8244226248566 R[8,4] - 384.7004189737583 R[8,5]
   - 454.655861940994 R[8,6] - 520.1644257027798 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 589.6362293231036 R[8,0] - 523.9147039737429 R[8,1]
   - 460.7815074561491 R[8,2] - 394.9055111031967 R[8,3]
   - 333.0919976561103 R[8,4] - 267.2160013072086 R[8,5]
   - 197.2605583399729 R[8,6] - 131.751994578187 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 600 Sold_0[8] + 600 Lost_0[8] - 600 Sold_1[8]
   + 400 Left_1[8] + 600 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction_for_second_order_amount[9] >= 0
 Fraction_upper_bound_9: Fraction_for_second_order_amount[9] <= 1
 Q0_upper_bound_9: - 2834.593301140667 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] + R[9,8] = 1
 Define_Q_hat_9: - 653.0097472313718 R[9,0] - 636.6681009027918 R[9,1]
   - 635.1530743486951 R[9,2] - 634.0903117361242 R[9,3]
   - 631.5446662209996 R[9,4] - 631.5446662250864 R[9,5]
   - 626.5018065331905 R[9,6] - 626.1266405157066 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 65.97066353697303 R[9,0] - 128.3637678731564 R[9,1]
   - 193.3921293971751 R[9,2] - 256.1203484144608 R[9,3]
   - 318.2279381469282 R[9,4] - 380.9561571714916 R[9,5]
   - 439.8960545253635 R[9,6] - 500.6702072783264 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 560.1559715584087 R[9,0] - 497.7628672222254 R[9,1]
   - 432.7345056982068 R[9,2] - 370.0062866809211 R[9,3]
   - 307.8986969484536 R[9,4] - 245.1704779238902 R[9,5]
   - 186.2305805700183 R[9,6] - 125.4564278170554 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 600 Sold_0[9] + 600 Lost_0[9] - 600 Sold_1[9]
   + 400 Left_1[9] + 600 Lost_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction_for_second_order_amount[10] >= 0
 Fraction_upper_bound_10: Fraction_for_second_order_amount[10] <= 1
 Q0_upper_bound_10:
   - 2834.593301140667 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 Ensure_only_one_R_true_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4]
   + R[10,5] + R[10,6] + R[10,7] + R[10,8] = 1
 Define_Q_hat_10: - 636.9310446296245 R[10,0] - 640.8851175537675 R[10,1]
   - 639.9132236358053 R[10,2] - 638.0772987135099 R[10,3]
   - 638.8443718412043 R[10,4] - 638.8443718447963 R[10,5]
   - 644.3280327808255 R[10,6] - 643.0426803990148 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 62.66010259684377 R[10,0] - 126.4482608552212 R[10,1]
   - 191.1042652436597 R[10,2] - 254.8127932325143 R[10,3]
   - 318.7070757842303 R[10,4] - 382.4156037756345 R[10,5]
   - 449.8284884350495 R[10,6] - 515.6256242550902 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 580.3825785248429 R[10,0] - 516.5944202664654 R[10,1]
   - 451.9384158780268 R[10,2] - 388.2298878891723 R[10,3]
   - 324.3356053374563 R[10,4] - 260.627077346052 R[10,5]
   - 193.214192686637 R[10,6] - 127.4170568665964 R[10,7] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: - Total_Demand_Up_to_K_minus_1[10] + Q0_var[10]
   + Lost_0_aux_10 = 0
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: - Total_Demand_from_k_to_T[10]
   + Q1_plus_left[10] + Lost_0_aux_10 = 0
 Profit_Constraint_10: - 600 Sold_0[10] + 600 Lost_0[10] - 600 Sold_1[10]
   + 400 Left_1[10] + 600 Lost_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction_for_second_order_amount[11] >= 0
 Fraction_upper_bound_11: Fraction_for_second_order_amount[11] <= 1
 Q0_upper_bound_11:
   - 2834.593301140667 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 Ensure_only_one_R_true_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4]
   + R[11,5] + R[11,6] + R[11,7] + R[11,8] = 1
 Define_Q_hat_11: - 642.5751777875416 R[11,0] - 629.92563920189 R[11,1]
   - 633.5230367161904 R[11,2] - 628.6502907376955 R[11,3]
   - 626.75628362211 R[11,4] - 626.7562836293387 R[11,5]
   - 629.0981253153926 R[11,6] - 625.2109975741638 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 63.82221419479438 R[11,0] - 125.6423616201859 R[11,1]
   - 184.3023283637682 R[11,2] - 246.272871160015 R[11,3]
   - 307.9635771102884 R[11,4] - 369.9341199118251 R[11,5]
   - 440.9141723098961 R[11,6] - 501.2699125772338 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 561.3887855714578 R[11,0] - 499.5686381460662 R[11,1]
   - 440.908671402484 R[11,2] - 378.9381286062371 R[11,3]
   - 317.2474226559637 R[11,4] - 255.2768798544271 R[11,5]
   - 184.2968274563561 R[11,6] - 123.9410871890184 R[11,7] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: - Total_Demand_Up_to_K_minus_1[11] + Q0_var[11]
   + Lost_0_aux_11 = 0
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: - Total_Demand_from_k_to_T[11]
   + Q1_plus_left[11] + Lost_0_aux_11 = 0
 Profit_Constraint_11: - 600 Sold_0[11] + 600 Lost_0[11] - 600 Sold_1[11]
   + 400 Left_1[11] + 600 Lost_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction_for_second_order_amount[12] >= 0
 Fraction_upper_bound_12: Fraction_for_second_order_amount[12] <= 1
 Q0_upper_bound_12:
   - 2834.593301140667 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 Ensure_only_one_R_true_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4]
   + R[12,5] + R[12,6] + R[12,7] + R[12,8] = 1
 Define_Q_hat_12: - 659.5983880694976 R[12,0] - 656.1889151544394 R[12,1]
   - 653.5161251739789 R[12,2] - 650.168680828725 R[12,3]
   - 656.2448512137976 R[12,4] - 656.2448512055195 R[12,5]
   - 653.9264611123996 R[12,6] - 654.4142948041751 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 67.32724668596853 R[12,0] - 132.4625082209494 R[12,1]
   - 201.1445747890543 R[12,2] - 265.7431336354104 R[12,3]
   - 332.7413404183825 R[12,4] - 397.3398992562176 R[12,5]
   - 459.0786285397936 R[12,6] - 525.2171808793313 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 587.0870459502823 R[12,0] - 521.9517844153015 R[12,1]
   - 453.2697178471967 R[12,2] - 388.6711590008405 R[12,3]
   - 321.6729522178683 R[12,4] - 257.0743933800333 R[12,5]
   - 195.3356640964573 R[12,6] - 129.1971117569196 R[12,7] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: - Total_Demand_Up_to_K_minus_1[12] + Q0_var[12]
   + Lost_0_aux_12 = 0
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: - Total_Demand_from_k_to_T[12]
   + Q1_plus_left[12] + Lost_0_aux_12 = 0
 Profit_Constraint_12: - 600 Sold_0[12] + 600 Lost_0[12] - 600 Sold_1[12]
   + 400 Left_1[12] + 600 Lost_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction_for_second_order_amount[13] >= 0
 Fraction_upper_bound_13: Fraction_for_second_order_amount[13] <= 1
 Q0_upper_bound_13:
   - 2834.593301140667 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 Ensure_only_one_R_true_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4]
   + R[13,5] + R[13,6] + R[13,7] + R[13,8] = 1
 Define_Q_hat_13: - 619.4582439018341 R[13,0] - 614.7789161510929 R[13,1]
   - 614.5969192977806 R[13,2] - 609.987978597713 R[13,3]
   - 608.5614651832327 R[13,4] - 608.5614651830655 R[13,5]
   - 605.1150916133047 R[13,6] - 602.9861234325407 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 59.06250066111303 R[13,0] - 119.5806600536675 R[13,1]
   - 179.9853665273544 R[13,2] - 240.4314895379031 R[13,3]
   - 301.5972205567401 R[13,4] - 362.0433435696749 R[13,5]
   - 421.816457935693 R[13,6] - 482.0938776012837 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 543.9236209686321 R[13,0] - 483.4054615760776 R[13,1]
   - 423.0007551023907 R[13,2] - 362.554632091842 R[13,3]
   - 301.388901073005 R[13,4] - 240.9427780600702 R[13,5]
   - 181.1696636940521 R[13,6] - 120.8922440284614 R[13,7] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: - Total_Demand_Up_to_K_minus_1[13] + Q0_var[13]
   + Lost_0_aux_13 = 0
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: - Total_Demand_from_k_to_T[13]
   + Q1_plus_left[13] + Lost_0_aux_13 = 0
 Profit_Constraint_13: - 600 Sold_0[13] + 600 Lost_0[13] - 600 Sold_1[13]
   + 400 Left_1[13] + 600 Lost_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction_for_second_order_amount[14] >= 0
 Fraction_upper_bound_14: Fraction_for_second_order_amount[14] <= 1
 Q0_upper_bound_14:
   - 2834.593301140667 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 Ensure_only_one_R_true_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4]
   + R[14,5] + R[14,6] + R[14,7] + R[14,8] = 1
 Define_Q_hat_14: - 666.8103027686508 R[14,0] - 666.9247203977643 R[14,1]
   - 668.3348708321591 R[14,2] - 676.2427232668474 R[14,3]
   - 674.0996214078865 R[14,4] - 674.0996214141434 R[14,5]
   - 675.3453161674978 R[14,6] - 675.6815660217019 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 68.81216022022132 R[14,0] - 135.2909483285155 R[14,1]
   - 201.0885630526116 R[14,2] - 269.0865218935036 R[14,3]
   - 335.3310129842486 R[14,4] - 403.3289718294587 R[14,5]
   - 473.935914413647 R[14,6] - 539.6856498649643 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 606.8694056854964 R[14,0] - 540.3906175772023 R[14,1]
   - 474.5930028531063 R[14,2] - 406.5950440122142 R[14,3]
   - 340.3505529214693 R[14,4] - 272.3525940762592 R[14,5]
   - 201.7456514920709 R[14,6] - 135.9959160407536 R[14,7] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: - Total_Demand_Up_to_K_minus_1[14] + Q0_var[14]
   + Lost_0_aux_14 = 0
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: - Total_Demand_from_k_to_T[14]
   + Q1_plus_left[14] + Lost_0_aux_14 = 0
 Profit_Constraint_14: - 600 Sold_0[14] + 600 Lost_0[14] - 600 Sold_1[14]
   + 400 Left_1[14] + 600 Lost_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction_for_second_order_amount[15] >= 0
 Fraction_upper_bound_15: Fraction_for_second_order_amount[15] <= 1
 Q0_upper_bound_15:
   - 2834.593301140667 Fraction_for_second_order_amount[15] + Q0_var[15]
   = 0
 Ensure_only_one_R_true_15: R[15,0] + R[15,1] + R[15,2] + R[15,3] + R[15,4]
   + R[15,5] + R[15,6] + R[15,7] + R[15,8] = 1
 Define_Q_hat_15: - 629.482073029855 R[15,0] - 620.3423432681907 R[15,1]
   - 620.9011013096087 R[15,2] - 615.3964046779361 R[15,3]
   - 620.9798281347767 R[15,4] - 620.9798281118942 R[15,5]
   - 620.9205533856983 R[15,6] - 619.2695134873584 R[15,7] + Q_hat[15] = 0
 Adjust_Q_hat_15: Q0_var[15] - Q_hat[15] + Q_hat_adjusted[15] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_15: Total_Demand_Up_to_K_minus_1[15]
   - 61.1263796943722 R[15,0] - 122.0524596350452 R[15,1]
   - 182.5255747742255 R[15,2] - 243.295139261883 R[15,3]
   - 308.9095005931928 R[15,4] - 369.6790650616155 R[15,5]
   - 434.5513074943055 R[15,6] - 497.7303866676676 R[15,7] = 0
 Constr_Total_Demand_from_K_to_T_15: Total_Demand_from_k_to_T[15]
   - 558.1431359349612 R[15,0] - 497.2170559942882 R[15,1]
   - 436.7439408551078 R[15,2] - 375.9743763674502 R[15,3]
   - 310.3600150361405 R[15,4] - 249.5904505677178 R[15,5]
   - 184.7182081350278 R[15,6] - 121.5391289616657 R[15,7] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: - Total_Demand_Up_to_K_minus_1[15] + Q0_var[15]
   + Lost_0_aux_15 = 0
 Constr_Q1_plus_left_15: - Left_0[15] + Q1_plus_left[15] - Q1_var[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: - Total_Demand_from_k_to_T[15]
   + Q1_plus_left[15] + Lost_0_aux_15 = 0
 Profit_Constraint_15: - 600 Sold_0[15] + 600 Lost_0[15] - 600 Sold_1[15]
   + 400 Left_1[15] + 600 Lost_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction_for_second_order_amount[16] >= 0
 Fraction_upper_bound_16: Fraction_for_second_order_amount[16] <= 1
 Q0_upper_bound_16:
   - 2834.593301140667 Fraction_for_second_order_amount[16] + Q0_var[16]
   = 0
 Ensure_only_one_R_true_16: R[16,0] + R[16,1] + R[16,2] + R[16,3] + R[16,4]
   + R[16,5] + R[16,6] + R[16,7] + R[16,8] = 1
 Define_Q_hat_16: - 656.188779803185 R[16,0] - 655.1504666021345 R[16,1]
   - 656.5143702799685 R[16,2] - 654.8325193943776 R[16,3]
   - 653.3252948610289 R[16,4] - 653.3252948448821 R[16,5]
   - 655.7239560759203 R[16,6] - 658.134241854963 R[16,7] + Q_hat[16] = 0
 Adjust_Q_hat_16: Q0_var[16] - Q_hat[16] + Q_hat_adjusted[16] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_16: Total_Demand_Up_to_K_minus_1[16]
   - 66.62521765867811 R[16,0] - 131.7549011661541 R[16,1]
   - 196.0068675080932 R[16,2] - 261.1952190078931 R[16,3]
   - 324.3646913485392 R[16,4] - 389.553042836913 R[16,5]
   - 459.4412919749529 R[16,6] - 527.7575403206475 R[16,7] = 0
 Constr_Total_Demand_from_K_to_T_16: Total_Demand_from_k_to_T[16]
   - 591.5090242954415 R[16,0] - 526.3793407879655 R[16,1]
   - 462.1273744460265 R[16,2] - 396.9390229462266 R[16,3]
   - 333.7695506055803 R[16,4] - 268.5811991172066 R[16,5]
   - 198.6929499791666 R[16,6] - 130.3767016334721 R[16,7] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: - Total_Demand_Up_to_K_minus_1[16] + Q0_var[16]
   + Lost_0_aux_16 = 0
 Constr_Q1_plus_left_16: - Left_0[16] + Q1_plus_left[16] - Q1_var[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: - Total_Demand_from_k_to_T[16]
   + Q1_plus_left[16] + Lost_0_aux_16 = 0
 Profit_Constraint_16: - 600 Sold_0[16] + 600 Lost_0[16] - 600 Sold_1[16]
   + 400 Left_1[16] + 600 Lost_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction_for_second_order_amount[17] >= 0
 Fraction_upper_bound_17: Fraction_for_second_order_amount[17] <= 1
 Q0_upper_bound_17:
   - 2834.593301140667 Fraction_for_second_order_amount[17] + Q0_var[17]
   = 0
 Ensure_only_one_R_true_17: R[17,0] + R[17,1] + R[17,2] + R[17,3] + R[17,4]
   + R[17,5] + R[17,6] + R[17,7] + R[17,8] = 1
 Define_Q_hat_17: - 670.0606810074385 R[17,0] - 643.7084514437744 R[17,1]
   - 642.8633687705703 R[17,2] - 641.2173612810172 R[17,3]
   - 638.4245439782155 R[17,4] - 638.4245439824796 R[17,5]
   - 639.3766121927372 R[17,6] - 638.4206969549257 R[17,7] + Q_hat[17] = 0
 Adjust_Q_hat_17: Q0_var[17] - Q_hat[17] + Q_hat_adjusted[17] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_17: Total_Demand_Up_to_K_minus_1[17]
   - 69.48140421984651 R[17,0] - 132.190483496478 R[17,1]
   - 197.6683374318843 R[17,2] - 260.736438421516 R[17,3]
   - 322.8504623198322 R[17,4] - 385.9185633170316 R[17,5]
   - 450.8355457984606 R[17,6] - 512.2845019339197 R[17,7] = 0
 Constr_Total_Demand_from_K_to_T_17: Total_Demand_from_k_to_T[17]
   - 568.9392915927409 R[17,0] - 506.2302123161095 R[17,1]
   - 440.7523583807032 R[17,2] - 377.6842573910714 R[17,3]
   - 315.5702334927552 R[17,4] - 252.5021324955558 R[17,5]
   - 187.5851500141268 R[17,6] - 126.1361938786677 R[17,7] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: - Total_Demand_Up_to_K_minus_1[17] + Q0_var[17]
   + Lost_0_aux_17 = 0
 Constr_Q1_plus_left_17: - Left_0[17] + Q1_plus_left[17] - Q1_var[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: - Total_Demand_from_k_to_T[17]
   + Q1_plus_left[17] + Lost_0_aux_17 = 0
 Profit_Constraint_17: - 600 Sold_0[17] + 600 Lost_0[17] - 600 Sold_1[17]
   + 400 Left_1[17] + 600 Lost_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction_for_second_order_amount[18] >= 0
 Fraction_upper_bound_18: Fraction_for_second_order_amount[18] <= 1
 Q0_upper_bound_18:
   - 2834.593301140667 Fraction_for_second_order_amount[18] + Q0_var[18]
   = 0
 Ensure_only_one_R_true_18: R[18,0] + R[18,1] + R[18,2] + R[18,3] + R[18,4]
   + R[18,5] + R[18,6] + R[18,7] + R[18,8] = 1
 Define_Q_hat_18: - 662.4974505374614 R[18,0] - 654.0914562314108 R[18,1]
   - 654.9497036336966 R[18,2] - 659.074416168052 R[18,3]
   - 664.4966139664058 R[18,4] - 664.4966139597898 R[18,5]
   - 658.1179191513743 R[18,6] - 657.527666809106 R[18,7] + Q_hat[18] = 0
 Adjust_Q_hat_18: Q0_var[18] - Q_hat[18] + Q_hat_adjusted[18] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_18: Total_Demand_Up_to_K_minus_1[18]
   - 67.92415572814414 R[18,0] - 132.5993565176183 R[18,1]
   - 197.3755171876719 R[18,2] - 263.1509466797384 R[18,3]
   - 332.6943432999489 R[18,4] - 398.4697727830095 R[18,5]
   - 462.9520201544292 R[18,6] - 525.9768116708046 R[18,7] = 0
 Constr_Total_Demand_from_K_to_T_18: Total_Demand_from_k_to_T[18]
   - 589.6035142272574 R[18,0] - 524.9283134377832 R[18,1]
   - 460.1521527677297 R[18,2] - 394.3767232756632 R[18,3]
   - 324.8333266554527 R[18,4] - 259.0578971723921 R[18,5]
   - 194.5756498009723 R[18,6] - 131.5508582845969 R[18,7] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: - Total_Demand_Up_to_K_minus_1[18] + Q0_var[18]
   + Lost_0_aux_18 = 0
 Constr_Q1_plus_left_18: - Left_0[18] + Q1_plus_left[18] - Q1_var[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: - Total_Demand_from_k_to_T[18]
   + Q1_plus_left[18] + Lost_0_aux_18 = 0
 Profit_Constraint_18: - 600 Sold_0[18] + 600 Lost_0[18] - 600 Sold_1[18]
   + 400 Left_1[18] + 600 Lost_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction_for_second_order_amount[19] >= 0
 Fraction_upper_bound_19: Fraction_for_second_order_amount[19] <= 1
 Q0_upper_bound_19:
   - 2834.593301140667 Fraction_for_second_order_amount[19] + Q0_var[19]
   = 0
 Ensure_only_one_R_true_19: R[19,0] + R[19,1] + R[19,2] + R[19,3] + R[19,4]
   + R[19,5] + R[19,6] + R[19,7] + R[19,8] = 1
 Define_Q_hat_19: - 677.9413670162061 R[19,0] - 689.3775835095641 R[19,1]
   - 688.6393331297101 R[19,2] - 691.6822929703638 R[19,3]
   - 689.7674216275063 R[19,4] - 689.7674216257237 R[19,5]
   - 685.5878447892753 R[19,6] - 681.5072268311986 R[19,7] + Q_hat[19] = 0
 Adjust_Q_hat_19: Q0_var[19] - Q_hat[19] + Q_hat_adjusted[19] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_19: Total_Demand_Up_to_K_minus_1[19]
   - 71.10401593972581 R[19,0] - 140.5738984433072 R[19,1]
   - 211.4990749378928 R[19,2] - 281.0934083474366 R[19,3]
   - 345.9609710239174 R[19,4] - 415.555304432028 R[19,5]
   - 480.713905265492 R[19,6] - 542.3185584653449 R[19,7] = 0
 Constr_Total_Demand_from_K_to_T_19: Total_Demand_from_k_to_T[19]
   - 610.4032096820425 R[19,0] - 540.9333271784612 R[19,1]
   - 470.0081506838756 R[19,2] - 400.4138172743318 R[19,3]
   - 335.546254597851 R[19,4] - 265.9519211897403 R[19,5]
   - 200.7933203562765 R[19,6] - 139.1886671564235 R[19,7] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: - Total_Demand_Up_to_K_minus_1[19] + Q0_var[19]
   + Lost_0_aux_19 = 0
 Constr_Q1_plus_left_19: - Left_0[19] + Q1_plus_left[19] - Q1_var[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: - Total_Demand_from_k_to_T[19]
   + Q1_plus_left[19] + Lost_0_aux_19 = 0
 Profit_Constraint_19: - 600 Sold_0[19] + 600 Lost_0[19] - 600 Sold_1[19]
   + 400 Left_1[19] + 600 Lost_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction_for_second_order_amount[20] >= 0
 Fraction_upper_bound_20: Fraction_for_second_order_amount[20] <= 1
 Q0_upper_bound_20:
   - 2834.593301140667 Fraction_for_second_order_amount[20] + Q0_var[20]
   = 0
 Ensure_only_one_R_true_20: R[20,0] + R[20,1] + R[20,2] + R[20,3] + R[20,4]
   + R[20,5] + R[20,6] + R[20,7] + R[20,8] = 1
 Define_Q_hat_20: - 657.3408537282514 R[20,0] - 663.9115407688039 R[20,1]
   - 659.5733249969705 R[20,2] - 661.6999644346117 R[20,3]
   - 656.7268247195998 R[20,4] - 656.7268247144395 R[20,5]
   - 660.6132553916959 R[20,6] - 660.0288700307156 R[20,7] + Q_hat[20] = 0
 Adjust_Q_hat_20: Q0_var[20] - Q_hat[20] + Q_hat_adjusted[20] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_20: Total_Demand_Up_to_K_minus_1[20]
   - 66.86242653248689 R[20,0] - 133.3055872705922 R[20,1]
   - 204.6111463076784 R[20,2] - 271.0668487503481 R[20,3]
   - 332.696343455997 R[20,4] - 399.1520459008618 R[20,5]
   - 461.7379183667337 R[20,6] - 527.1174659813375 R[20,7] = 0
 Constr_Total_Demand_from_K_to_T_20: Total_Demand_from_k_to_T[20]
   - 593.1664427368588 R[20,0] - 526.7232819987535 R[20,1]
   - 455.4177229616673 R[20,2] - 388.9620205189976 R[20,3]
   - 327.3325258133488 R[20,4] - 260.8768233684839 R[20,5]
   - 198.290950902612 R[20,6] - 132.9114032880081 R[20,7] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: - Total_Demand_Up_to_K_minus_1[20] + Q0_var[20]
   + Lost_0_aux_20 = 0
 Constr_Q1_plus_left_20: - Left_0[20] + Q1_plus_left[20] - Q1_var[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: - Total_Demand_from_k_to_T[20]
   + Q1_plus_left[20] + Lost_0_aux_20 = 0
 Profit_Constraint_20: - 600 Sold_0[20] + 600 Lost_0[20] - 600 Sold_1[20]
   + 400 Left_1[20] + 600 Lost_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction_for_second_order_amount[21] >= 0
 Fraction_upper_bound_21: Fraction_for_second_order_amount[21] <= 1
 Q0_upper_bound_21:
   - 2834.593301140667 Fraction_for_second_order_amount[21] + Q0_var[21]
   = 0
 Ensure_only_one_R_true_21: R[21,0] + R[21,1] + R[21,2] + R[21,3] + R[21,4]
   + R[21,5] + R[21,6] + R[21,7] + R[21,8] = 1
 Define_Q_hat_21: - 623.0249592957899 R[21,0] - 617.9686038503864 R[21,1]
   - 618.3296459988147 R[21,2] - 611.066563373143 R[21,3]
   - 613.324535801714 R[21,4] - 613.324535803246 R[21,5]
   - 612.910860687015 R[21,6] - 611.2927030283214 R[21,7] + Q_hat[21] = 0
 Adjust_Q_hat_21: Q0_var[21] - Q_hat[21] + Q_hat_adjusted[21] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_21: Total_Demand_Up_to_K_minus_1[21]
   - 59.79687761678263 R[21,0] - 120.6489014375003 R[21,1]
   - 180.9275949851553 R[21,2] - 241.2812193430238 R[21,3]
   - 304.0546522739444 R[21,4] - 364.4082766321808 R[21,5]
   - 427.8246529836387 R[21,6] - 490.5854549025652 R[21,7] = 0
 Constr_Total_Demand_from_K_to_T_21: Total_Demand_from_k_to_T[21]
   - 551.4958242091408 R[21,0] - 490.6438003884232 R[21,1]
   - 430.3651068407682 R[21,2] - 370.0114824828996 R[21,3]
   - 307.238049551979 R[21,4] - 246.8844251937426 R[21,5]
   - 183.4680488422847 R[21,6] - 120.7072469233582 R[21,7] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: - Total_Demand_Up_to_K_minus_1[21] + Q0_var[21]
   + Lost_0_aux_21 = 0
 Constr_Q1_plus_left_21: - Left_0[21] + Q1_plus_left[21] - Q1_var[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: - Total_Demand_from_k_to_T[21]
   + Q1_plus_left[21] + Lost_0_aux_21 = 0
 Profit_Constraint_21: - 600 Sold_0[21] + 600 Lost_0[21] - 600 Sold_1[21]
   + 400 Left_1[21] + 600 Lost_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction_for_second_order_amount[22] >= 0
 Fraction_upper_bound_22: Fraction_for_second_order_amount[22] <= 1
 Q0_upper_bound_22:
   - 2834.593301140667 Fraction_for_second_order_amount[22] + Q0_var[22]
   = 0
 Ensure_only_one_R_true_22: R[22,0] + R[22,1] + R[22,2] + R[22,3] + R[22,4]
   + R[22,5] + R[22,6] + R[22,7] + R[22,8] = 1
 Define_Q_hat_22: - 630.499828502049 R[22,0] - 646.5097233557708 R[22,1]
   - 643.5768583096824 R[22,2] - 640.655945240903 R[22,3]
   - 637.5313981699298 R[22,4] - 637.5313981623996 R[22,5]
   - 636.1679506992095 R[22,6] - 636.2972940720222 R[22,7] + Q_hat[22] = 0
 Adjust_Q_hat_22: Q0_var[22] - Q_hat[22] + Q_hat_adjusted[22] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_22: Total_Demand_Up_to_K_minus_1[22]
   - 61.33593276568047 R[22,0] - 126.2962109744843 R[22,1]
   - 193.559565940491 R[22,2] - 257.8635903377857 R[22,3]
   - 318.6500458075897 R[22,4] - 382.9540702029292 R[22,5]
   - 442.2363358467373 R[22,6] - 507.6892428627151 R[22,7] = 0
 Constr_Total_Demand_from_K_to_T_22: Total_Demand_from_k_to_T[22]
   - 574.9613643299523 R[22,0] - 510.0010861211484 R[22,1]
   - 442.7377311551417 R[22,2] - 378.4337067578469 R[22,3]
   - 317.6472512880431 R[22,4] - 253.3432268927035 R[22,5]
   - 194.0609612488954 R[22,6] - 128.6080542329176 R[22,7] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: - Total_Demand_Up_to_K_minus_1[22] + Q0_var[22]
   + Lost_0_aux_22 = 0
 Constr_Q1_plus_left_22: - Left_0[22] + Q1_plus_left[22] - Q1_var[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: - Total_Demand_from_k_to_T[22]
   + Q1_plus_left[22] + Lost_0_aux_22 = 0
 Profit_Constraint_22: - 600 Sold_0[22] + 600 Lost_0[22] - 600 Sold_1[22]
   + 400 Left_1[22] + 600 Lost_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction_for_second_order_amount[23] >= 0
 Fraction_upper_bound_23: Fraction_for_second_order_amount[23] <= 1
 Q0_upper_bound_23:
   - 2834.593301140667 Fraction_for_second_order_amount[23] + Q0_var[23]
   = 0
 Ensure_only_one_R_true_23: R[23,0] + R[23,1] + R[23,2] + R[23,3] + R[23,4]
   + R[23,5] + R[23,6] + R[23,7] + R[23,8] = 1
 Define_Q_hat_23: - 650.0228063673405 R[23,0] - 644.1515792923711 R[23,1]
   - 643.3115465367871 R[23,2] - 649.6769762816381 R[23,3]
   - 645.721893021831 R[23,4] - 645.7218930077506 R[23,5]
   - 640.4750989073705 R[23,6] - 635.2689031187927 R[23,7] + Q_hat[23] = 0
 Adjust_Q_hat_23: Q0_var[23] - Q_hat[23] + Q_hat_adjusted[23] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_23: Total_Demand_Up_to_K_minus_1[23]
   - 65.35566057312303 R[23,0] - 129.0528203644701 R[23,1]
   - 194.2025384583327 R[23,2] - 259.3027069867662 R[23,3]
   - 323.13897370585 R[23,4] - 388.2391422283883 R[23,5]
   - 448.6612896567469 R[23,6] - 505.0685672169014 R[23,7] = 0
 Constr_Total_Demand_from_K_to_T_23: Total_Demand_from_k_to_T[23]
   - 569.9132450048778 R[23,0] - 506.2160852135307 R[23,1]
   - 441.0663671196681 R[23,2] - 375.9661985912347 R[23,3]
   - 312.1299318721508 R[23,4] - 247.0297633496125 R[23,5]
   - 186.6076159212539 R[23,6] - 130.2003383610994 R[23,7] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: - Total_Demand_Up_to_K_minus_1[23] + Q0_var[23]
   + Lost_0_aux_23 = 0
 Constr_Q1_plus_left_23: - Left_0[23] + Q1_plus_left[23] - Q1_var[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: - Total_Demand_from_k_to_T[23]
   + Q1_plus_left[23] + Lost_0_aux_23 = 0
 Profit_Constraint_23: - 600 Sold_0[23] + 600 Lost_0[23] - 600 Sold_1[23]
   + 400 Left_1[23] + 600 Lost_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction_for_second_order_amount[24] >= 0
 Fraction_upper_bound_24: Fraction_for_second_order_amount[24] <= 1
 Q0_upper_bound_24:
   - 2834.593301140667 Fraction_for_second_order_amount[24] + Q0_var[24]
   = 0
 Ensure_only_one_R_true_24: R[24,0] + R[24,1] + R[24,2] + R[24,3] + R[24,4]
   + R[24,5] + R[24,6] + R[24,7] + R[24,8] = 1
 Define_Q_hat_24: - 647.2776449908281 R[24,0] - 665.4102593671287 R[24,1]
   - 665.2370053168274 R[24,2] - 655.6140948132324 R[24,3]
   - 653.4743066457901 R[24,4] - 653.4743066586045 R[24,5]
   - 647.6172127872279 R[24,6] - 649.3483282835896 R[24,7] + Q_hat[24] = 0
 Adjust_Q_hat_24: Q0_var[24] - Q_hat[24] + Q_hat_adjusted[24] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_24: Total_Demand_Up_to_K_minus_1[24]
   - 64.79043934523328 R[24,0] - 131.9288464682745 R[24,1]
   - 198.8455100175742 R[24,2] - 264.2072849577532 R[24,3]
   - 324.1252624296961 R[24,4] - 389.487037378786 R[24,5]
   - 451.3011411392334 R[24,6] - 518.6247750629275 R[24,7] = 0
 Constr_Total_Demand_from_K_to_T_24: Total_Demand_from_k_to_T[24]
   - 584.5578861337044 R[24,0] - 517.4194790106633 R[24,1]
   - 450.5028154613636 R[24,2] - 385.1410405211845 R[24,3]
   - 325.2230630492417 R[24,4] - 259.8612881001517 R[24,5]
   - 198.0471843397044 R[24,6] - 130.7235504160103 R[24,7] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: - Total_Demand_Up_to_K_minus_1[24] + Q0_var[24]
   + Lost_0_aux_24 = 0
 Constr_Q1_plus_left_24: - Left_0[24] + Q1_plus_left[24] - Q1_var[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: - Total_Demand_from_k_to_T[24]
   + Q1_plus_left[24] + Lost_0_aux_24 = 0
 Profit_Constraint_24: - 600 Sold_0[24] + 600 Lost_0[24] - 600 Sold_1[24]
   + 400 Left_1[24] + 600 Lost_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction_for_second_order_amount[25] >= 0
 Fraction_upper_bound_25: Fraction_for_second_order_amount[25] <= 1
 Q0_upper_bound_25:
   - 2834.593301140667 Fraction_for_second_order_amount[25] + Q0_var[25]
   = 0
 Ensure_only_one_R_true_25: R[25,0] + R[25,1] + R[25,2] + R[25,3] + R[25,4]
   + R[25,5] + R[25,6] + R[25,7] + R[25,8] = 1
 Define_Q_hat_25: - 653.9703258311727 R[25,0] - 662.995928853235 R[25,1]
   - 666.2282165152479 R[25,2] - 666.7618557989204 R[25,3]
   - 663.3040073232678 R[25,4] - 663.3040073249679 R[25,5]
   - 658.6743617818076 R[25,6] - 658.0386744303999 R[25,7] + Q_hat[25] = 0
 Adjust_Q_hat_25: Q0_var[25] - Q_hat[25] + Q_hat_adjusted[25] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_25: Total_Demand_Up_to_K_minus_1[25]
   - 66.16844404645764 R[25,0] - 132.6233911689587 R[25,1]
   - 195.7995732273561 R[25,2] - 262.6134983331168 R[25,3]
   - 325.5783498770606 R[25,4] - 392.3922749831307 R[25,5]
   - 460.5300102532573 R[25,6] - 524.4108217520557 R[25,7] = 0
 Constr_Total_Demand_from_K_to_T_25: Total_Demand_from_k_to_T[25]
   - 591.8702300054897 R[25,0] - 525.4152828829885 R[25,1]
   - 462.2391008245913 R[25,2] - 395.4251757188306 R[25,3]
   - 332.4603241748867 R[25,4] - 265.6463990688165 R[25,5]
   - 197.50866379869 R[25,6] - 133.6278522998917 R[25,7] = 0
 Constr_Left_0_diff_aux_25: Sold_0[25] - Q0_var[25] + Left_0_aux_25 = 0
 Constr_Lost_0_diff_aux_25: - Total_Demand_Up_to_K_minus_1[25] + Q0_var[25]
   + Lost_0_aux_25 = 0
 Constr_Q1_plus_left_25: - Left_0[25] + Q1_plus_left[25] - Q1_var[25] = 0
 Constr_Left_1_diff_aux_25: Sold_1[25] - Q1_plus_left[25] + Left_0_aux_25
   = 0
 Constr_Lost_1_diff_aux_25: - Total_Demand_from_k_to_T[25]
   + Q1_plus_left[25] + Lost_0_aux_25 = 0
 Profit_Constraint_25: - 600 Sold_0[25] + 600 Lost_0[25] - 600 Sold_1[25]
   + 400 Left_1[25] + 600 Lost_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction_for_second_order_amount[26] >= 0
 Fraction_upper_bound_26: Fraction_for_second_order_amount[26] <= 1
 Q0_upper_bound_26:
   - 2834.593301140667 Fraction_for_second_order_amount[26] + Q0_var[26]
   = 0
 Ensure_only_one_R_true_26: R[26,0] + R[26,1] + R[26,2] + R[26,3] + R[26,4]
   + R[26,5] + R[26,6] + R[26,7] + R[26,8] = 1
 Define_Q_hat_26: - 645.7983662069047 R[26,0] - 631.4520244301187 R[26,1]
   - 631.5477108131109 R[26,2] - 632.6823982874737 R[26,3]
   - 630.0944894884184 R[26,4] - 630.0944894702498 R[26,5]
   - 633.5038911546108 R[26,6] - 635.8887172072795 R[26,7] + Q_hat[26] = 0
 Adjust_Q_hat_26: Q0_var[26] - Q_hat[26] + Q_hat_adjusted[26] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_26: Total_Demand_Up_to_K_minus_1[26]
   - 64.48585988488014 R[26,0] - 126.3962299495762 R[26,1]
   - 188.9782806383638 R[26,2] - 251.7542416225497 R[26,3]
   - 314.6333187541504 R[26,4] - 377.4092797288532 R[26,5]
   - 444.2220213705872 R[26,6] - 510.3367998473163 R[26,7] = 0
 Constr_Total_Demand_from_K_to_T_26: Total_Demand_from_k_to_T[26]
   - 571.4028598943501 R[26,0] - 509.4924898296541 R[26,1]
   - 446.9104391408664 R[26,2] - 384.1344781566805 R[26,3]
   - 321.2554010250798 R[26,4] - 258.479440050377 R[26,5]
   - 191.666698408643 R[26,6] - 125.551919931914 R[26,7] = 0
 Constr_Left_0_diff_aux_26: Sold_0[26] - Q0_var[26] + Left_0_aux_26 = 0
 Constr_Lost_0_diff_aux_26: - Total_Demand_Up_to_K_minus_1[26] + Q0_var[26]
   + Lost_0_aux_26 = 0
 Constr_Q1_plus_left_26: - Left_0[26] + Q1_plus_left[26] - Q1_var[26] = 0
 Constr_Left_1_diff_aux_26: Sold_1[26] - Q1_plus_left[26] + Left_0_aux_26
   = 0
 Constr_Lost_1_diff_aux_26: - Total_Demand_from_k_to_T[26]
   + Q1_plus_left[26] + Lost_0_aux_26 = 0
 Profit_Constraint_26: - 600 Sold_0[26] + 600 Lost_0[26] - 600 Sold_1[26]
   + 400 Left_1[26] + 600 Lost_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction_for_second_order_amount[27] >= 0
 Fraction_upper_bound_27: Fraction_for_second_order_amount[27] <= 1
 Q0_upper_bound_27:
   - 2834.593301140667 Fraction_for_second_order_amount[27] + Q0_var[27]
   = 0
 Ensure_only_one_R_true_27: R[27,0] + R[27,1] + R[27,2] + R[27,3] + R[27,4]
   + R[27,5] + R[27,6] + R[27,7] + R[27,8] = 1
 Define_Q_hat_27: - 627.5720974535485 R[27,0] - 616.5155267052855 R[27,1]
   - 614.7484525007932 R[27,2] - 616.1689999683659 R[27,3]
   - 616.1542642398138 R[27,4] - 616.1542642569372 R[27,5]
   - 615.0910257257106 R[27,6] - 616.0919283956205 R[27,7] + Q_hat[27] = 0
 Adjust_Q_hat_27: Q0_var[27] - Q_hat[27] + Q_hat_adjusted[27] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_27: Total_Demand_Up_to_K_minus_1[27]
   - 60.73312094109232 R[27,0] - 121.1499933542727 R[27,1]
   - 183.5786358135947 R[27,2] - 244.8672414526365 R[27,3]
   - 308.8736044605618 R[27,4] - 370.1622101146124 R[27,5]
   - 430.029009323604 R[27,6] - 493.5147210783186 R[27,7] = 0
 Constr_Total_Demand_from_K_to_T_27: Total_Demand_from_k_to_T[27]
   - 555.3588103840342 R[27,0] - 494.9419379708538 R[27,1]
   - 432.5132955115319 R[27,2] - 371.22468987249 R[27,3]
   - 307.2183268645647 R[27,4] - 245.9297212105141 R[27,5]
   - 186.0629220015225 R[27,6] - 122.5772102468079 R[27,7] = 0
 Constr_Left_0_diff_aux_27: Sold_0[27] - Q0_var[27] + Left_0_aux_27 = 0
 Constr_Lost_0_diff_aux_27: - Total_Demand_Up_to_K_minus_1[27] + Q0_var[27]
   + Lost_0_aux_27 = 0
 Constr_Q1_plus_left_27: - Left_0[27] + Q1_plus_left[27] - Q1_var[27] = 0
 Constr_Left_1_diff_aux_27: Sold_1[27] - Q1_plus_left[27] + Left_0_aux_27
   = 0
 Constr_Lost_1_diff_aux_27: - Total_Demand_from_k_to_T[27]
   + Q1_plus_left[27] + Lost_0_aux_27 = 0
 Profit_Constraint_27: - 600 Sold_0[27] + 600 Lost_0[27] - 600 Sold_1[27]
   + 400 Left_1[27] + 600 Lost_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction_for_second_order_amount[28] >= 0
 Fraction_upper_bound_28: Fraction_for_second_order_amount[28] <= 1
 Q0_upper_bound_28:
   - 2834.593301140667 Fraction_for_second_order_amount[28] + Q0_var[28]
   = 0
 Ensure_only_one_R_true_28: R[28,0] + R[28,1] + R[28,2] + R[28,3] + R[28,4]
   + R[28,5] + R[28,6] + R[28,7] + R[28,8] = 1
 Define_Q_hat_28: - 661.1293153654609 R[28,0] - 646.6334217325765 R[28,1]
   - 647.9316138615188 R[28,2] - 646.3525819651547 R[28,3]
   - 645.3949755918337 R[28,4] - 645.394975590777 R[28,5]
   - 648.397481235522 R[28,6] - 645.0456365622587 R[28,7] + Q_hat[28] = 0
 Adjust_Q_hat_28: Q0_var[28] - Q_hat[28] + Q_hat_adjusted[28] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_28: Total_Demand_Up_to_K_minus_1[28]
   - 67.64246043385158 R[28,0] - 131.2173055509955 R[28,1]
   - 194.6215795202568 R[28,2] - 258.5483362857657 R[28,3]
   - 322.0564110867531 R[28,4] - 385.9831678522737 R[28,5]
   - 455.8253842412699 R[28,6] - 517.1921259137548 R[28,7] = 0
 Constr_Total_Demand_from_K_to_T_28: Total_Demand_from_k_to_T[28]
   - 577.403169904489 R[28,0] - 513.8283247873451 R[28,1]
   - 450.4240508180839 R[28,2] - 386.4972940525749 R[28,3]
   - 322.9892192515875 R[28,4] - 259.0624624860669 R[28,5]
   - 189.2202460970708 R[28,6] - 127.8535044245858 R[28,7] = 0
 Constr_Left_0_diff_aux_28: Sold_0[28] - Q0_var[28] + Left_0_aux_28 = 0
 Constr_Lost_0_diff_aux_28: - Total_Demand_Up_to_K_minus_1[28] + Q0_var[28]
   + Lost_0_aux_28 = 0
 Constr_Q1_plus_left_28: - Left_0[28] + Q1_plus_left[28] - Q1_var[28] = 0
 Constr_Left_1_diff_aux_28: Sold_1[28] - Q1_plus_left[28] + Left_0_aux_28
   = 0
 Constr_Lost_1_diff_aux_28: - Total_Demand_from_k_to_T[28]
   + Q1_plus_left[28] + Lost_0_aux_28 = 0
 Profit_Constraint_28: - 600 Sold_0[28] + 600 Lost_0[28] - 600 Sold_1[28]
   + 400 Left_1[28] + 600 Lost_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction_for_second_order_amount[29] >= 0
 Fraction_upper_bound_29: Fraction_for_second_order_amount[29] <= 1
 Q0_upper_bound_29:
   - 2834.593301140667 Fraction_for_second_order_amount[29] + Q0_var[29]
   = 0
 Ensure_only_one_R_true_29: R[29,0] + R[29,1] + R[29,2] + R[29,3] + R[29,4]
   + R[29,5] + R[29,6] + R[29,7] + R[29,8] = 1
 Define_Q_hat_29: - 685.9187206921865 R[29,0] - 695.83295404183 R[29,1]
   - 695.1001803077377 R[29,2] - 694.4079777797511 R[29,3]
   - 701.2175393944985 R[29,4] - 701.217539402915 R[29,5]
   - 705.7248068399848 R[29,6] - 704.5037664381184 R[29,7] + Q_hat[29] = 0
 Adjust_Q_hat_29: Q0_var[29] - Q_hat[29] + Q_hat_adjusted[29] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_29: Total_Demand_Up_to_K_minus_1[29]
   - 72.74653126819688 R[29,0] - 142.857290432263 R[29,1]
   - 214.6911293543464 R[29,2] - 284.2551865834201 R[29,3]
   - 353.607102484312 R[29,4] - 423.1711597125133 R[29,5]
   - 496.1116229263203 R[29,6] - 565.3756526601494 R[29,7] = 0
 Constr_Total_Demand_from_K_to_T_29: Total_Demand_from_k_to_T[29]
   - 631.7572348472798 R[29,0] - 561.6464756832138 R[29,1]
   - 489.8126367611303 R[29,2] - 420.2485795320566 R[29,3]
   - 350.8966636311647 R[29,4] - 281.3326064029634 R[29,5]
   - 208.3921431891565 R[29,6] - 139.1281134553274 R[29,7] = 0
 Constr_Left_0_diff_aux_29: Sold_0[29] - Q0_var[29] + Left_0_aux_29 = 0
 Constr_Lost_0_diff_aux_29: - Total_Demand_Up_to_K_minus_1[29] + Q0_var[29]
   + Lost_0_aux_29 = 0
 Constr_Q1_plus_left_29: - Left_0[29] + Q1_plus_left[29] - Q1_var[29] = 0
 Constr_Left_1_diff_aux_29: Sold_1[29] - Q1_plus_left[29] + Left_0_aux_29
   = 0
 Constr_Lost_1_diff_aux_29: - Total_Demand_from_k_to_T[29]
   + Q1_plus_left[29] + Lost_0_aux_29 = 0
 Profit_Constraint_29: - 600 Sold_0[29] + 600 Lost_0[29] - 600 Sold_1[29]
   + 400 Left_1[29] + 600 Lost_1[29] + profits_vars[29] = 0
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
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
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
 Max_Constraint_15: Q1_var[15] = MAX ( Q_hat_adjusted[15] , 0 )
 Constr_Sold_0_min_15: Sold_0[15] = MIN (
   Total_Demand_Up_to_K_minus_1[15] , Q0_var[15] )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Total_Demand_from_k_to_T[15] ,
   Q1_plus_left[15] )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_0_aux_15 , 0 )
 Max_Constraint_16: Q1_var[16] = MAX ( Q_hat_adjusted[16] , 0 )
 Constr_Sold_0_min_16: Sold_0[16] = MIN (
   Total_Demand_Up_to_K_minus_1[16] , Q0_var[16] )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Total_Demand_from_k_to_T[16] ,
   Q1_plus_left[16] )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_0_aux_16 , 0 )
 Max_Constraint_17: Q1_var[17] = MAX ( Q_hat_adjusted[17] , 0 )
 Constr_Sold_0_min_17: Sold_0[17] = MIN (
   Total_Demand_Up_to_K_minus_1[17] , Q0_var[17] )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Total_Demand_from_k_to_T[17] ,
   Q1_plus_left[17] )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_0_aux_17 , 0 )
 Max_Constraint_18: Q1_var[18] = MAX ( Q_hat_adjusted[18] , 0 )
 Constr_Sold_0_min_18: Sold_0[18] = MIN (
   Total_Demand_Up_to_K_minus_1[18] , Q0_var[18] )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Total_Demand_from_k_to_T[18] ,
   Q1_plus_left[18] )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_0_aux_18 , 0 )
 Max_Constraint_19: Q1_var[19] = MAX ( Q_hat_adjusted[19] , 0 )
 Constr_Sold_0_min_19: Sold_0[19] = MIN (
   Total_Demand_Up_to_K_minus_1[19] , Q0_var[19] )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Total_Demand_from_k_to_T[19] ,
   Q1_plus_left[19] )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_0_aux_19 , 0 )
 Max_Constraint_20: Q1_var[20] = MAX ( Q_hat_adjusted[20] , 0 )
 Constr_Sold_0_min_20: Sold_0[20] = MIN (
   Total_Demand_Up_to_K_minus_1[20] , Q0_var[20] )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Total_Demand_from_k_to_T[20] ,
   Q1_plus_left[20] )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_0_aux_20 , 0 )
 Max_Constraint_21: Q1_var[21] = MAX ( Q_hat_adjusted[21] , 0 )
 Constr_Sold_0_min_21: Sold_0[21] = MIN (
   Total_Demand_Up_to_K_minus_1[21] , Q0_var[21] )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Total_Demand_from_k_to_T[21] ,
   Q1_plus_left[21] )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_0_aux_21 , 0 )
 Max_Constraint_22: Q1_var[22] = MAX ( Q_hat_adjusted[22] , 0 )
 Constr_Sold_0_min_22: Sold_0[22] = MIN (
   Total_Demand_Up_to_K_minus_1[22] , Q0_var[22] )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Total_Demand_from_k_to_T[22] ,
   Q1_plus_left[22] )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_0_aux_22 , 0 )
 Max_Constraint_23: Q1_var[23] = MAX ( Q_hat_adjusted[23] , 0 )
 Constr_Sold_0_min_23: Sold_0[23] = MIN (
   Total_Demand_Up_to_K_minus_1[23] , Q0_var[23] )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Total_Demand_from_k_to_T[23] ,
   Q1_plus_left[23] )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_0_aux_23 , 0 )
 Max_Constraint_24: Q1_var[24] = MAX ( Q_hat_adjusted[24] , 0 )
 Constr_Sold_0_min_24: Sold_0[24] = MIN (
   Total_Demand_Up_to_K_minus_1[24] , Q0_var[24] )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Total_Demand_from_k_to_T[24] ,
   Q1_plus_left[24] )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_0_aux_24 , 0 )
 Max_Constraint_25: Q1_var[25] = MAX ( Q_hat_adjusted[25] , 0 )
 Constr_Sold_0_min_25: Sold_0[25] = MIN (
   Total_Demand_Up_to_K_minus_1[25] , Q0_var[25] )
 Constr_Left_0_max_25: Left_0[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_0_max_25: Lost_0[25] = MAX ( Lost_0_aux_25 , 0 )
 Constr_Sold_1_min_25: Sold_1[25] = MIN ( Total_Demand_from_k_to_T[25] ,
   Q1_plus_left[25] )
 Constr_Left_1_max_25: Left_1[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_1_max_25: Lost_1[25] = MAX ( Lost_0_aux_25 , 0 )
 Max_Constraint_26: Q1_var[26] = MAX ( Q_hat_adjusted[26] , 0 )
 Constr_Sold_0_min_26: Sold_0[26] = MIN (
   Total_Demand_Up_to_K_minus_1[26] , Q0_var[26] )
 Constr_Left_0_max_26: Left_0[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_0_max_26: Lost_0[26] = MAX ( Lost_0_aux_26 , 0 )
 Constr_Sold_1_min_26: Sold_1[26] = MIN ( Total_Demand_from_k_to_T[26] ,
   Q1_plus_left[26] )
 Constr_Left_1_max_26: Left_1[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_1_max_26: Lost_1[26] = MAX ( Lost_0_aux_26 , 0 )
 Max_Constraint_27: Q1_var[27] = MAX ( Q_hat_adjusted[27] , 0 )
 Constr_Sold_0_min_27: Sold_0[27] = MIN (
   Total_Demand_Up_to_K_minus_1[27] , Q0_var[27] )
 Constr_Left_0_max_27: Left_0[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_0_max_27: Lost_0[27] = MAX ( Lost_0_aux_27 , 0 )
 Constr_Sold_1_min_27: Sold_1[27] = MIN ( Total_Demand_from_k_to_T[27] ,
   Q1_plus_left[27] )
 Constr_Left_1_max_27: Left_1[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_1_max_27: Lost_1[27] = MAX ( Lost_0_aux_27 , 0 )
 Max_Constraint_28: Q1_var[28] = MAX ( Q_hat_adjusted[28] , 0 )
 Constr_Sold_0_min_28: Sold_0[28] = MIN (
   Total_Demand_Up_to_K_minus_1[28] , Q0_var[28] )
 Constr_Left_0_max_28: Left_0[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_0_max_28: Lost_0[28] = MAX ( Lost_0_aux_28 , 0 )
 Constr_Sold_1_min_28: Sold_1[28] = MIN ( Total_Demand_from_k_to_T[28] ,
   Q1_plus_left[28] )
 Constr_Left_1_max_28: Left_1[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_1_max_28: Lost_1[28] = MAX ( Lost_0_aux_28 , 0 )
 Max_Constraint_29: Q1_var[29] = MAX ( Q_hat_adjusted[29] , 0 )
 Constr_Sold_0_min_29: Sold_0[29] = MIN (
   Total_Demand_Up_to_K_minus_1[29] , Q0_var[29] )
 Constr_Left_0_max_29: Left_0[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_0_max_29: Lost_0[29] = MAX ( Lost_0_aux_29 , 0 )
 Constr_Sold_1_min_29: Sold_1[29] = MIN ( Total_Demand_from_k_to_T[29] ,
   Q1_plus_left[29] )
 Constr_Left_1_max_29: Left_1[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_1_max_29: Lost_1[29] = MAX ( Lost_0_aux_29 , 0 )
End
