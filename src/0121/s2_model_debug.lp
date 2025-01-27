\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: 0.3629370792343273 alphas[0] - 1.380816439025236 alphas[1]
   + 0.1456017588547462 alphas[2] + 0.6508025075686793 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 1453.252517908159 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 1068.993602307392
 R5: Q0_var[0] + Q_hat_adjusted_0 = 1348.747552561108
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 279.7539502868945
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 500 Sold_0[0] - 500 Sold_1[0] + 500 Lost_0[0] + 500 Lost_1[0]
   + 500 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.5389518359619722 alphas[0] - 0.7587744334102131 alphas[1]
   - 0.2125425940893099 alphas[2] + 0.8179904522965766 alphas[3]
   - alphas[4] + f_var[1] = 0
 R13: - 1453.252517908159 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 1052.096134848146
 R15: Q0_var[1] + Q_hat_adjusted_1 = 1370.612111769512
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 318.5159768996707
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 500 Sold_0[1] - 500 Sold_1[1] + 500 Lost_0[1] + 500 Lost_1[1]
   + 500 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: 0.0704860532973403 alphas[0] + 0.4116864373552839 alphas[1]
   - 1.416454709959565 alphas[2] - 1.060351956133714 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 1453.252517908159 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1532.785321129661
 R25: Q0_var[2] + Q_hat_adjusted_2 = 1981.600919045505
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 448.8155979636325
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 500 Sold_0[2] - 500 Sold_1[2] + 500 Lost_0[2] + 500 Lost_1[2]
   + 500 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: 0.3842425308077874 alphas[0] - 0.6942014022851634 alphas[1]
   + 1.528221828396284 alphas[2] + 0.7223183878646142 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 1453.252517908159 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 633.2812822562447
 R35: Q0_var[3] + Q_hat_adjusted_3 = 763.3940180427094
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 130.1127357955089
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 500 Sold_0[3] - 500 Sold_1[3] + 500 Lost_0[3] + 500 Lost_1[3]
   + 500 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 1.041396362027678 alphas[0] + 1.60130056329161 alphas[1]
   - 1.273867487915398 alphas[2] + 0.5213967003607619 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 1453.252517908159 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 2199.416354561477
 R45: Q0_var[4] + Q_hat_adjusted_4 = 2632.799712339834
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 433.3833577228339
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 500 Sold_0[4] - 500 Sold_1[4] + 500 Lost_0[4] + 500 Lost_1[4]
   + 500 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.1633166932572295 alphas[0] - 0.2068265727639536 alphas[1]
   - 1.29175138263669 alphas[2] - 0.0708562603421244 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 1453.252517908159 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1756.633821495953
 R55: Q0_var[5] + Q_hat_adjusted_5 = 2191.952756302805
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 435.3189348090797
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 500 Sold_0[5] - 500 Sold_1[5] + 500 Lost_0[5] + 500 Lost_1[5]
   + 500 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.9658716573538862 alphas[0] + 1.514372419818931 alphas[1]
   + 0.8345285628671657 alphas[2] - 1.62687302729598 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 1453.252517908159 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 942.8041280121209
 R65: Q0_var[6] + Q_hat_adjusted_6 = 1147.995408466625
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 205.1912805369205
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 500 Sold_0[6] - 500 Sold_1[6] + 500 Lost_0[6] + 500 Lost_1[6]
   + 500 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.496174957986087 alphas[0] - 1.263141220857213 alphas[1]
   + 1.210838876139294 alphas[2] + 0.8636588488600513 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 1453.252517908159 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 469.6081939327933
 R75: Q0_var[7] + Q_hat_adjusted_7 = 634.0713428049906
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 164.4631488408105
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 500 Sold_0[7] - 500 Sold_1[7] + 500 Lost_0[7] + 500 Lost_1[7]
   + 500 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.885874379848093 alphas[0] + 0.202436046779539 alphas[1]
   + 0.3492397412676438 alphas[2] - 1.685194843566105 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 1453.252517908159 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 870.8868638528354
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1128.601039437272
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 257.7141754620919
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 500 Sold_0[8] - 500 Sold_1[8] + 500 Lost_0[8] + 500 Lost_1[8]
   + 500 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 1.259384184332354 alphas[0] + 0.5739646010964145 alphas[1]
   + 0.1261854070758312 alphas[2] + 0.867109190387246 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 1453.252517908159 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 1050.894932896207
 R95: Q0_var[9] + Q_hat_adjusted_9 = 1332.750318320855
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 281.8553854713113
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 500 Sold_0[9] - 500 Sold_1[9] + 500 Lost_0[9] + 500 Lost_1[9]
   + 500 Left_1[9] + profits_vars[9] = 0
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
 Q0_var[0] <= 1454.252517908159
 Q0_var[1] <= 1454.252517908159
 Q0_var[2] <= 1454.252517908159
 Q0_var[3] <= 1454.252517908159
 Q0_var[4] <= 1454.252517908159
 Q0_var[5] <= 1454.252517908159
 Q0_var[6] <= 1454.252517908159
 Q0_var[7] <= 1454.252517908159
 Q0_var[8] <= 1454.252517908159
 Q0_var[9] <= 1454.252517908159
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
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1068.993602307392 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 279.7539502868945 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 1052.096134848146 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 318.5159768996707 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1532.785321129661 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 448.8155979636325 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 633.2812822562447 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 130.1127357955089 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 2199.416354561477 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 433.3833577228339 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1756.633821495953 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 435.3189348090797 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 942.8041280121209 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 205.1912805369205 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 469.6081939327933 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 164.4631488408105 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 870.8868638528354 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 257.7141754620919 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 1050.894932896207 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 281.8553854713113 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
End
