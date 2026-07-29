
import pandas as pd
import matplotlib.pyplot as plt

# Data for the "My" category
my_data = {
    # 'Total cache size': [3.125, 12.5, 21.875, 31.25, 40.625],
    # 'Total system cost': [2255385.0, 2008195, 1822515.0, 1721570, 1588290],
    # 'Overall cache hit rate': [0.160, 0.292, 0.367, 0.421, 0.468],
    'Total cache size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Total system cost': [2984451.0/17012*15000, 2781373.0/17122*15000, 2656583.0/17345*15000, 2407366.0/16442*15000, 2355192.0/16954*15000, 2268755.0/16665*15000, 2170000.0/16758*15000, 2149862.0/17010*15000, 1977310.0/16190*15000, 2003645.0/16863*15000],
    'Total Request': [17012, 17122, 17345, 16442, 16954, 16665, 16758, 17010, 16190, 16863],
    'Overall cache hit rate': [0.2397258745005246, 0.30553464688952847, 0.3571676802780191, 0.39299255401763283, 0.4303747010364071, 0.4500358969756798, 0.482602489182317, 0.4961277725368026, 0.5195758189179241, 0.5369325694138387],
    'Overall average latency': [0.9062, 0.8954, 0.8885, 0.8829, 0.8773, 0.8732, 0.8695, 0.8653, 0.8628, 0.8599],
    'Overall average ISL hops': [3.2222222222222223, 3.0872052299789865,  2.9854436229205175, 2.8821957736215693, 2.7974096958174903, 2.785380663241476, 2.7152802359882005, 2.620175230878522, 2.601441172816499, 2.544361081338573],
    'Total cache miss cost':[1702900/17012*15000,
                            1568450/17122*15000,
                            1479800/17345*15000,
                            1328800/16442*15000,
                            1286100/16954*15000,
                            1225650/16665*15000,
                            1159850/16758*15000,
                            1148350/17010*15000,
                            1044250/16190*15000,
                            1050700/16863*15000
                            ],
}
my_df = pd.DataFrame(my_data)


mobile_data = {
    'Total cache size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Total system cost': [3151201.0/16887*15000,
                        3020456.0/17098*15000,
                        2855657.0/16879*15000,
                        2674036.0/16501*15000,
                        2617061.0/16595*15000,
                        2556464.0/16630*15000,
                        2483330.0/16597*15000,
                        2404745.0/16442*15000,
                        2399384.0/16696*15000,
                        2343054.0/16757*15000
                        ],
    'Total Request': [16887, 
                    17098,
                    16879,
                    16501,
                    16595,
                    16630,
                    16597,
                    16442,
                    16696,
                    16757
                    ],
    'Overall cache hit rate': [0.17501238348268564,
                            0.22708379394207384,
                            0.2657417527267861,
                            0.29913142857142855,
                            0.32388252051821204,
                            0.347898441540918,
                            0.3655248718585589,
                            0.3843738623953404,
                            0.3982990312067503,
                            0.41818666548177114
                            ],
    'Overall average latency': [0.918, 0.908, 0.904, 0.896, 0.895, 0.891, 0.888, 0.885, 0.883, 0.881],
    'Overall average ISL hops': [3.600557546794106,
                                3.5861581920903953,
                                3.4993300580616347,
                                3.4742006964229186,
                                3.4143451809157863,
                                3.4351644841903544,
                                3.3688939715732724,
                                3.3767225635065583,
                                3.3390939597315437,
                                3.3577667408802876
                                ],
    'Total cache miss cost':[1832050/16887*15000,
                            1747950/17098*15000,
                            1645950/16879*15000,
                            1533150/16501*15000,
                            1495200/16595*15000,
                            1449850/16630*15000,
                            1404950/16597*15000,
                            1352900/16442*15000,
                            1347750/16696*15000,
                            1309400/16757*15000
                            ],
}
mobile_df = pd.DataFrame(mobile_data)

# RFP_data = {
#     'Total cache size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
#     'Total system cost': [3080316.0/16877*15000,
#                         2829988.0/16522*15000,
#                         2725946.0/16921*15000,
#                         2591937.0/16857*15000,
#                         2470868.0/16813*15000,
#                         2397461.0/16695*15000,
#                         2301537.0/16609*15000,
#                         2161653.0/15954*15000,
#                         2214296.0/16750*15000,
#                         2148112.0/16343*15000
#                         ],
#     'Total Request': [16877,
#                     16522,
#                     16921,
#                     16857,
#                     16813,
#                     16695,
#                     16609,
#                     15954,
#                     16750,
#                     16343
#                     ],
#     'Overall cache hit rate': [0.2054243179583343,
#                             0.27188919862156763,
#                             0.3092090369900847,
#                             0.36997483164410583,
#                             0.40409725861951096,
#                             0.42631062975584705,
#                             0.4453971921094722,
#                             0.46413868813466286,
#                             0.46796676199698245,
#                             0.48686927635123
#                             ],
#     'Overall average latency': [0.9138, 0.9056, 0.8970, 0.8859, 0.8790, 0.8736, 0.8709, 0.8684, 0.8646, 0.8633],
#     'Overall average ISL hops': [3.5981440150358277,
#                                 3.5812791093945244,
#                                 3.545776727702304,
#                                 3.608825283243888,
#                                 3.6922161390145205,
#                                 3.5991825265040234,
#                                 3.487632931055084,
#                                 3.5265855221012172,
#                                 3.3880979864050276,
#                                 3.2879249112125826
#                                 ],
#     'Total cache miss cost':[35139*50/16877*15000,
#                             31713*50/16522*15000,
#                             29978*50/16921*15000,
#                             27802*50/16857*15000,
#                             26421*50/16813*15000,
#                             25364*50/16695*15000,
#                             24287*50/16609*15000,
#                             22934*50/15954*15000,
#                             23421*50/16750*15000,
#                             22561*50/16343*15000
#                             ],                       
# }

RFP_data = {
    'Total cache size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Total system cost': [3006470.0/16517*15000,
                        2801583.0/16234*15000,
                        2744337.0/16761*15000,
                        2486516.0/15857*15000,
                        2479023.0/16531*15000,
                        2454724.0/16860*15000,
                        2373674.0/16574*15000,
                        2222023.0/15888*15000,
                        2329015.0/17028*15000,
                        2245063.0/16912*15000
                        ],
    'Total Request': [16517,
                    16234,
                    16761,
                    15857,
                    16531,
                    16860,
                    16574,
                    15888,
                    17028,
                    16912
                    ],
    'Overall cache hit rate': [0.2095954368775731,
                            0.2647305808257523,
                            0.3120688877868446,
                            0.35049340149803826,
                            0.38771321718507706,
                            0.4104534117227864,
                            0.42316591806179393,
                            0.4389847285067873,
                            0.45346046286305686,
                            0.4679979632048529
                            ],
    'Overall average latency': [0.9138, 0.9056, 0.8970, 0.8859, 0.8790, 0.8736, 0.8709, 0.8684, 0.8646, 0.8633],
    'Overall average ISL hops': [3.6080009908347783,
                                3.604083665338645,
                                3.5691443915537655,
                                3.5958279009126466,
                                3.5849197461739455,
                                3.5453068366454192,
                                3.6134119758943455,
                                3.50498429605353,
                                3.4220403022670025,
                                3.2822979291917167
                                ],
    'Total cache miss cost':[34366*50/16517*15000,
                            31521*50/16234*15000,
                            30518*50/16761*15000,
                            27315*50/15857*15000,
                            26850*50/16531*15000,
                            26473*50/16860*15000,
                            25428*50/16574*15000,
                            23805*50/15888*15000,
                            24820*50/17028*15000,
                            24030*50/16912*15000
                            ],                       
}
RFP_df = pd.DataFrame(RFP_data)

SCA_data = {
    'Total cache size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Total system cost': [3249574.0/16958*15000,
                        3006329.0/16531*15000,
                        3004102.0/17086*15000,
                        2838379.0/16456*15000,
                        2815582.0/16874*15000,
                        2749029.0/16819*15000,
                        2676904.0/16436*15000,
                        2761216.0/17265*15000,
                        2687828.0/17067*15000,
                        2643762.0/16917*15000
                        ],
    'Total Request': [16958,
                    16531,
                    17086,
                    16456,
                    16874,
                    16819,
                    16436,
                    17265,
                    17067,
                    16917
                    ], 
    'Overall cache hit rate': [0.14945364821995066,
                            0.19047293914151758,
                            0.2330567081604426,
                            0.2495494056717848,
                            0.2626135443834877,
                            0.2837506248011996,
                            0.2998427672955975,
                            0.30385440927981,
                            0.31504233388551445,
                            0.3305023653976121
                            ],
    'Overall average latency': [0.9216, 0.9152, 0.9108, 0.9077, 0.9054, 0.9025, 0.9011, 0.9006, 0.9004, 0.8997],
    'Overall average ISL hops': [3.5932452276064613,
                                3.542314572726022,
                                3.4897988098611505,
                                3.521210366295063,
                                3.3928470455188013,
                                3.293788637032676,
                                3.286201274640581,
                                3.271389972981087,
                                3.167983883465055,
                                3.177370493789761
                                ],
    'Total cache miss cost':[1895800/16958*15000,
                            1746750/16531*15000,
                            1739350/17086*15000,
                            1637550/16456*15000,
                            1618100/16874*15000,
                            1578300/16819*15000,
                            1529150/16436*15000,
                            1583000/17265*15000,
                            1534850/17067*15000,
                            1509850/16917*15000
                            ],
}
SCA_df = pd.DataFrame(SCA_data)

# Plot 1: Total Cache Size vs. Total System Cost
plt.plot(my_df['Total cache size'], my_df['Total system cost'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Total cache size'], mobile_df['Total system cost'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Total cache size'], RFP_df['Total system cost'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Total cache size'], SCA_df['Total system cost'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Cache size(% of total content size)', fontsize=16)
plt.ylabel('Total System Cost', fontsize=16)
plt.title('Total Cache Size vs. Total System Cost', fontsize=16)
plt.xticks(my_df['Total cache size'], fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_cost.png')
plt.clf()

# Plot 2: Total Cache Size vs. Overall Cache Hit Rate
plt.plot(my_df['Total cache size'], my_df['Overall cache hit rate'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Total cache size'], mobile_df['Overall cache hit rate'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Total cache size'], RFP_df['Overall cache hit rate'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Total cache size'], SCA_df['Overall cache hit rate'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Cache size(% of total content size)', fontsize=16)
plt.ylabel('Hit rate', fontsize=16)
plt.title('Total Cache Size vs. Overall Cache Hit Rate', fontsize=16)
plt.xticks(my_df['Total cache size'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.ylim(0.1, 0.6)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_hit_rate.png')
plt.clf()

# Plot 3: Total Cache Size vs. Overall Average Latency
plt.plot(my_df['Total cache size'], my_df['Overall average latency'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Total cache size'], mobile_df['Overall average latency'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Total cache size'], RFP_df['Overall average latency'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Total cache size'], SCA_df['Overall average latency'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Cache size(% of total content size)', fontsize=16)
plt.ylabel('Average Latency (s)', fontsize=16)
plt.title('Total Cache Size vs. Overall Average Latency', fontsize=16)
plt.xticks(my_df['Total cache size'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_latency.png')
plt.clf()   

# Plot 4: Total Cache Size vs. Overall Average ISL Hops
plt.plot(my_df['Total cache size'], my_df['Overall average ISL hops'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Total cache size'], mobile_df['Overall average ISL hops'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Total cache size'], RFP_df['Overall average ISL hops'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Total cache size'], SCA_df['Overall average ISL hops'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Cache size(% of total content size)', fontsize=16)
plt.ylabel('Average ISL Hops', fontsize=16)
plt.title('Total Cache Size vs. Overall Average ISL Hops', fontsize=16)
plt.xticks(my_df['Total cache size'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14, loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_ISL_hops.png')
plt.clf()

# Plot 5: Total Cache Size vs. Total Cache Miss Cost
plt.plot(my_df['Total cache size'], my_df['Total cache miss cost'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Total cache size'], mobile_df['Total cache miss cost'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Total cache size'], RFP_df['Total cache miss cost'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Total cache size'], SCA_df['Total cache miss cost'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Cache size(% of total content size)')
plt.ylabel('Total Cache Miss Cost')
plt.title('Total Cache Size vs. Total Cache Miss Cost')
plt.xticks(my_df['Total cache size'])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_miss_cost.png')
plt.clf()


# Plot 1: Total Cache Size vs. Total System Cost (Column Plot)
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = range(len(my_df['Total cache size']))
width = 0.2
ax.bar([x - 1.5*width for x in x_pos], my_df['Total system cost'], width, label='AMSCC', alpha=0.8)
ax.bar([x - 0.5*width for x in x_pos], mobile_df['Total system cost'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([x + 0.5*width for x in x_pos], RFP_df['Total system cost'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([x + 1.5*width for x in x_pos], SCA_df['Total system cost'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('Cache size(% of total content size)')
ax.set_ylabel('Total System Cost')
ax.set_title('Total Cache Size vs. Total System Cost')
ax.set_xticks(x_pos)
ax.set_xticklabels(my_df['Total cache size'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_cost_column.png', dpi=300, bbox_inches='tight')
plt.clf()

# Plot 2: Total Cache Size vs. Overall Cache Hit Rate (Column Plot)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([x - 1.5*width for x in x_pos], my_df['Overall cache hit rate'], width, label='AMSCC', alpha=0.8)
ax.bar([x - 0.5*width for x in x_pos], mobile_df['Overall cache hit rate'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([x + 0.5*width for x in x_pos], RFP_df['Overall cache hit rate'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([x + 1.5*width for x in x_pos], SCA_df['Overall cache hit rate'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('Cache size(% of total content size)')
ax.set_ylabel('Hit rate')
ax.set_title('Total Cache Size vs. Overall Cache Hit Rate')
ax.set_xticks(x_pos)
ax.set_xticklabels(my_df['Total cache size'])
ax.set_ylim(0, 0.6)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_hit_rate_column.png', dpi=300, bbox_inches='tight')
plt.clf()

# Plot 3: Total Cache Size vs. Overall Average Latency (Column Plot)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([x - 1.5*width for x in x_pos], my_df['Overall average latency'], width, label='AMSCC', alpha=0.8)
ax.bar([x - 0.5*width for x in x_pos], mobile_df['Overall average latency'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([x + 0.5*width for x in x_pos], RFP_df['Overall average latency'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([x + 1.5*width for x in x_pos], SCA_df['Overall average latency'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('Cache size(% of total content size)')
ax.set_ylabel('Average Latency (s)')
ax.set_title('Total Cache Size vs. Overall Average Latency')
ax.set_xticks(x_pos)
ax.set_xticklabels(my_df['Total cache size'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_latency_column.png', dpi=300, bbox_inches='tight')
plt.clf()

# Plot 4: Total Cache Size vs. Overall Average ISL Hops (Column Plot)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([x - 1.5*width for x in x_pos], my_df['Overall average ISL hops'], width, label='AMSCC', alpha=0.8)
ax.bar([x - 0.5*width for x in x_pos], mobile_df['Overall average ISL hops'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([x + 0.5*width for x in x_pos], RFP_df['Overall average ISL hops'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([x + 1.5*width for x in x_pos], SCA_df['Overall average ISL hops'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('Cache size(% of total content size)')
ax.set_ylabel('Average ISL Hops')
ax.set_title('Total Cache Size vs. Overall Average ISL Hops')
ax.set_xticks(x_pos)
ax.set_xticklabels(my_df['Total cache size'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/cache_vs_ISL_hops_column.png', dpi=300, bbox_inches='tight')
plt.clf()


print("Finished plotting simulation results.")