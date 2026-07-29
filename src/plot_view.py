
import pandas as pd
import matplotlib.pyplot as plt

# Data for the "My" category
my_data = {

    'Number of views': [4, 8, 12, 16, 20, 24, 28, 32],
    'Total system cost (|V|)': [1540244.0/16657*15000, 2184410.0/17186*15000, 2363340.0/16636*15000, 2656583.0/17345*15000, 2742115.0/16934*15000, 2745387.0/16627*15000, 2932618.0/17258*15000, 2933164.0/16799*15000],
    'Overall cache hit rate (|V|)': [0.5592230334756854,
                                     0.4424542236185028,
                                     0.398255477661322,
                                     0.3571676802780191,
                                     0.32175675973467904,
                                     0.31644869326606034,
                                     0.2978804659156005,
                                     0.28333225339640167],
    'Total Request': [16657, 17186, 16636, 17345, 16934, 16627, 17258, 16799],
    'ISL hops': [2.2109448712764523,
                 2.651174168297456,
                 2.8595451246068233,
                 2.9854436229205175,
                 3.054942528735632,
                 3.1194842067904887,
                 3.1662928907826866,
                 3.220793993857354],
    'Average service latency': [0.8565511706685636,
                                0.8806172913619582,
                                0.890090299018732,
                                0.8953444138463511,
                                0.8996906508253645,
                                0.9020137926820571,
                                0.9059759525975455,
                                0.9075833141585686],
}
my_df = pd.DataFrame(my_data)

mobile_data = {


    'Number of views': [4, 8, 12, 16, 20, 24, 28, 32],
    'Total system cost (|V|)': [1859930.0/16926*15000,
                                2323407.0/16194*15000,
                                2615732.0/16377*15000,
                                2855657.0/16879*15000,
                                2940328.0/16668*15000,
                                3028420.0/16672*15000,
                                3107233.0/16779*15000,
                                3116245.0/16521*15000],
    'Overall cache hit rate (|V|)': [0.4385007870596537,
                                     0.34517602661502556,
                                     0.2968426987408382,
                                     0.2657417527267861,
                                     0.24247297418029126,
                                     0.22501881613317395,
                                     0.2134566716118335,
                                     0.20392779210474113],
    'Total Request': [16926, 16194, 16377, 16879, 16668, 16672, 16779, 16521],
    'ISL hops': [2.6850632009969733,
                 3.2418542156533423,
                 3.3699036323202374,
                 3.4993300580616347,
                 3.600630282194528,
                 3.6998573466476463,
                 3.681939940786691,
                 3.7637993835808348],
    'Average service latency': [0.8779,
                                0.8977,
                                0.9041,
                                0.909042878875668,
                                0.9142,
                                0.9146,
                                0.9176,
                                0.9192],

}
mobile_df = pd.DataFrame(mobile_data)

RFP_data = {


    'Number of views': [4, 8, 12, 16, 20, 24, 28, 32], # 20~32 problem
    ###################################################################################################
    'Total system cost (|V|)': [1735684.0/16890*15000,
                                2273087.0/16618*15000,
                                2504120.0/16416*15000,
                                2744337.0/16761*15000,
                                2830108.0/16641*15000,
                                2940118.0/16522*15000,
                                3004635.0/16542*15000,
                                3033848.0/16611*15000],
    'Overall cache hit rate (|V|)': [0.499065574671577,
                                     0.396984069074547,
                                     0.35005414057718565,
                                     0.3120688877868446,
                                     0.285435148792813,
                                     0.2564349344196249,
                                     0.24769503546099292,
                                     0.24675695252310192],
    'Total Request': [16890, 16618, 16416, 16761, 16641, 16522, 16542, 16611],
    'ISL hops': [2.878504672897196,
                 3.2621583472346978,
                 3.4187317321133563,
                 3.5691443915537655,
                 3.5954620920709764,
                 3.6686796140222304,
                 3.695916180497678,
                 3.779090583601862],
    'Average service latency': [0.8437455005860836,
                                0.8771094618390709,
                                0.8839444761308201,
                                0.905069601030598,
                                0.9058575434899759,
                                0.9075047860616656,
                                0.9106419592383578,
                                0.912356197501032],
}
RFP_df = pd.DataFrame(RFP_data)

SCA_data = {

    'Number of views': [4, 8, 12, 16, 20, 24, 28, 32],
    'Total system cost (|V|)': [1927687.0/16206*15000,
                                2507332.0/16571*15000,
                                2770602.0/16500*15000,
                                3004102.0/17086*15000,
                                2984472.0/16324*15000,
                                3121512.0/16566*15000,
                                3220530.0/16776*15000,
                                3259499.0/16877*15000], 
    'Overall cache hit rate (|V|)': [0.3779829098911149,
                                     0.3012462502743702,
                                     0.25039259346068204,
                                     0.23001837136722814,
                                     0.20859035869316883,
                                     0.19175335813414715,
                                     0.1795174661504473,
                                     0.18019712099597976],
    'Total Request': [16206, 16571, 16500, 17086, 16324, 16566, 16776, 16877],
    'ISL hops': [2.641135236792614,
                 3.1638876370737568,
                 3.3488846770666276,
                 3.4912672907642865,
                 3.5226187131216635,
                 3.638233680746023,
                 3.687728813559322,
                 3.688078244779276],
    'Average service latency': [0.8907,
                                0.9065,
                                0.9130,
                                0.9174,
                                0.9204,
                                0.9228,
                                0.9234,
                                0.9261],
}
SCA_df = pd.DataFrame(SCA_data)


# Plot 5: Number of Views vs. Total System Cost
plt.rcParams.update({'font.size': 16})
plt.plot(my_df['Number of views'], [x/1000 for x in my_df['Total system cost (|V|)']], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Number of views'], [x/1000 for x in mobile_df['Total system cost (|V|)']], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Number of views'], [x/1000 for x in RFP_df['Total system cost (|V|)']], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Number of views'], [x/1000 for x in SCA_df['Total system cost (|V|)']], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Number of Views', fontsize=16)
plt.ylabel('Total System Cost (K)', fontsize=16)
plt.title('Number of Views vs. Total System Cost', fontsize=16)
plt.xticks(my_df['Number of views'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/views_vs_cost.png')
plt.clf()

# Plot 6: Number of Views vs. Overall Cache Hit Rate
plt.plot(my_df['Number of views'], my_df['Overall cache hit rate (|V|)'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Number of views'], mobile_df['Overall cache hit rate (|V|)'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Number of views'], RFP_df['Overall cache hit rate (|V|)'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Number of views'], SCA_df['Overall cache hit rate (|V|)'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Number of Views', fontsize=16)
plt.ylabel('Hit rate', fontsize=16)
plt.title('Number of Views vs. Overall Cache Hit Rate', fontsize=16)
plt.xticks(my_df['Number of views'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.ylim(0.1, 0.6)
# plt.yticks([i/5 for i in range(6)])
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/views_vs_hit_rate.png')
plt.clf()

# Plot 7 : Number of Views vs. ISL Hops
plt.plot(my_df['Number of views'], my_df['ISL hops'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Number of views'], mobile_df['ISL hops'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Number of views'], RFP_df['ISL hops'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Number of views'], SCA_df['ISL hops'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Number of Views', fontsize=16)
plt.ylabel('ISL Hops', fontsize=16)
plt.title('Number of Views vs. ISL Hops', fontsize=16)
plt.xticks(my_df['Number of views'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/views_vs_ISL_hops.png')
plt.clf()

# Plot 8: Number of Views vs. Average Service Latency
plt.plot(my_df['Number of views'], my_df['Average service latency'], marker='o', linestyle='-', label='AMSCC')
plt.plot(mobile_df['Number of views'], mobile_df['Average service latency'], marker='s', linestyle='--', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['Number of views'], RFP_df['Average service latency'], marker='^', linestyle='-.', label='RFP + EAYB')
plt.plot(SCA_df['Number of views'], SCA_df['Average service latency'], marker='D', linestyle=':', label='SCA + EAYB')
plt.xlabel('Number of Views', fontsize=16)
plt.ylabel('Average Service Latency (s)', fontsize=16)
plt.title('Number of Views vs. Average Service Latency', fontsize=16)
plt.xticks(my_df['Number of views'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/views_vs_latency.png')
plt.clf()

# Column plot for Total System Cost
plt.figure(figsize=(10, 6))
x_pos = range(len(my_df['Number of views']))
width = 0.2

plt.bar([x - 1.5*width for x in x_pos], [x/1000 for x in my_df['Total system cost (|V|)']], width, label='AMSCC', alpha=0.8)
plt.bar([x - 0.5*width for x in x_pos], [x/1000 for x in mobile_df['Total system cost (|V|)']], width, label='Mobile Proxy (EAYB)', alpha=0.8)
plt.bar([x + 0.5*width for x in x_pos], [x/1000 for x in RFP_df['Total system cost (|V|)']], width, label='RFP + EAYB', alpha=0.8)
plt.bar([x + 1.5*width for x in x_pos], [x/1000 for x in SCA_df['Total system cost (|V|)']], width, label='SCA + EAYB', alpha=0.8)

plt.xlabel('Number of Views', fontsize=16)
plt.ylabel('Total System Cost (K)', fontsize=16)
plt.title('Total System Cost Comparison', fontsize=16)
plt.xticks(x_pos, my_df['Number of views'])
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/column_cost_comparison.png')
plt.clf()

# Column plot for Cache Hit Rate
plt.figure(figsize=(10, 6))

plt.bar([x - 1.5*width for x in x_pos], my_df['Overall cache hit rate (|V|)'], width, label='AMSCC', alpha=0.8)
plt.bar([x - 0.5*width for x in x_pos], mobile_df['Overall cache hit rate (|V|)'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
plt.bar([x + 0.5*width for x in x_pos], RFP_df['Overall cache hit rate (|V|)'], width, label='RFP + EAYB', alpha=0.8)
plt.bar([x + 1.5*width for x in x_pos], SCA_df['Overall cache hit rate (|V|)'], width, label='SCA + EAYB', alpha=0.8)

plt.xlabel('Number of Views', fontsize=14)
plt.ylabel('Hit Rate', fontsize=14)
plt.title('Cache Hit Rate Comparison', fontsize=16)
plt.xticks(x_pos, my_df['Number of views'])
plt.ylim(0, 0.6)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/column_hitrate_comparison.png')
plt.clf()

print("Finished plotting simulation results.")
