
import pandas as pd
import matplotlib.pyplot as plt

# Data for the "My" category
my_data = {

    'DIBR constraints': [1, 2, 3, 4],
    'Total system cost': [4174685.0/16287*15000, 2737889.0/16648*15000, 2528561.0/16529*15000, 2130090.0/16855*15000],
    'Total system cost (DIBR)': [4174685.0/16287*15000, 2737889.0/16648*15000, 2528561.0/16529*15000, 2130090.0/16855*15000],
    'DIBR cost': [0.00, 77679.0/16648*15000, 92091.0/16529*15000, 173865.0/16855*15000],
    'Network Transmission cost': [4175725.00/16287*15000, 2660240/16648*15000, 2436760/16529*15000, 1956225/16855*15000],
    'Overall cache hit rate (DIBR)': [0.3094498574650045,
                                      0.3354498725756987,
                                      0.3549842025733779,
                                      0.40481684835945997],
    'Total Request': [16287, 16648, 16529, 16855],  
    'ISL hops':[4.963273118794114, 3.2238123167155424, 2.946398659966499, 2.4093688845401173],
    'Average Latency': [0.8969, 0.89724, 0.89534, 0.894028],
}
my_df = pd.DataFrame(my_data)

mobile_data = {

    'DIBR constraints': [1, 2, 3, 4],
    'Total system cost': [4697745.0/16816*15000, 3027589.0/16987*15000, 2859927.0/16966*15000, 2348718.0/16887*15000],
    'Total system cost (DIBR)': [4697745.0/16816*15000, 3027589.0/16987*15000, 2859927.0/16966*15000, 2348718.0/16887*15000],
    'DIBR cost': [0.00, 79809.0/16987*15000, 92517.0/16966*15000, 174993.0/16887*15000],
    'Network Transmission cost': [4697745.0/16816*15000, 2947780.0/16987*15000, 2767410.0/16966*15000, 2173725.0/16887*15000],
    'Overall cache hit rate (DIBR)': [0.22977368486062885,
                                      0.25336990262363396,
                                      0.264489522960321,
                                      0.3164711827442219],
    'Total Request': [16816, 16987, 16966, 16887],  
    'ISL hops':[6.000591891092039, 3.765680473372781, 3.4609004739336493, 2.7897252090800477],
    'Average Latency': [0.9109, 0.9095, 0.909042, 0.9080],
}
mobile_df = pd.DataFrame(mobile_data)

RFP_data = {

    'DIBR constraints': [1, 2, 3, 4],
    'Total system cost': [4536840.0/16673*15000, 2885616.0/16574*15000, 2770590.0/16814*15000, 2240790.0/16434*15000],
    'Total system cost (DIBR)': [4536840.0/16673*15000, 2885616.0/16574*15000, 2770590.0/16814*15000, 2240790.0/16434*15000],
    'DIBR cost': [0, 38380.5/16574*15000, 56406.0/16814*15000, 182788.5/16434*15000],
    'Network Transmission cost': [4536840.0/16673*15000, 2847235.5/16574*15000, 2714184.0/16814*15000, 2058001.5/16434*15000],
    'Overall cache hit rate (DIBR)': [0.27111400038728595,
                                      0.29386602974704157,
                                      0.3064733648010789,
                                      0.3431592349636026],
    'Total Request': [16673, 16574, 16814, 16434],  
    'ISL hops':[6.037441110835606, 3.8319189360354655, 3.606605504587156, 2.7944808467741935],
    'Average Latency': [0.8961721861720834, 0.8977285942234953, 0.8989132287980599, 0.8986677864219215],
}
RFP_df = pd.DataFrame(RFP_data)

SCA_data = {

    'DIBR constraints': [1, 2, 3, 4],
    'Total system cost': [4990070.0/17288*15000, 3176913.0/16982*15000, 3004102.0/17086*15000, 2476030.0/16992*15000],
    'Total system cost (DIBR)': [4990070.0/17288*15000, 3176913.0/16982*15000, 3004102.0/17086*15000, 2476030.0/16992*15000],
    'DIBR cost': [0, 79518.0/16982*15000, 93537.0/17086*15000, 181065.0/16992*15000],
    'Network Transmission cost': [4990210/17288*15000, 3097735/16982*15000, 2910855/17086*15000, 2295105/16992*15000],
    'Overall cache hit rate (DIBR)': [0.19884795713328868, 
                                      0.2109519654372486,
                                      0.23001837136722814, 
                                      0.271541723337867],
    'Total Request': [17288, 16982, 17086, 16992],  
    'ISL hops':[5.949972360420122, 3.7540339553809456, 3.4912672907642865, 2.7283761160714284],
    'Average Latency': [0.9179, 0.9167, 0.9174, 0.9174],
}
SCA_df = pd.DataFrame(SCA_data)


# Plot 1: DIBR Constraints vs. Total System Cost
plt.plot(my_df['DIBR constraints'], my_df['Total system cost'], marker='o', label='My Algo')
plt.plot(mobile_df['DIBR constraints'], mobile_df['Total system cost'], marker='s', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['DIBR constraints'], RFP_df['Total system cost'], marker='^', label='RFP + EAYB')
plt.plot(SCA_df['DIBR constraints'], SCA_df['Total system cost'], marker='D', label='SCA + EAYB')
plt.xlabel('DIBR Constraints', fontsize=16)
plt.ylabel('Total System Cost', fontsize=16)
plt.title('DIBR Constraints vs. Total System Cost', fontsize=16)
plt.xticks(my_df['DIBR constraints'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_cost.png')
plt.clf()   

# Plot 7: DIBR Constraints vs. Total System Cost
plt.plot(my_df['DIBR constraints'], my_df['Total system cost (DIBR)'], marker='o', label='My Algo')
plt.plot(mobile_df['DIBR constraints'], mobile_df['Total system cost (DIBR)'], marker='s', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['DIBR constraints'], RFP_df['Total system cost (DIBR)'], marker='^', label='RFP + EAYB')
plt.plot(SCA_df['DIBR constraints'], SCA_df['Total system cost (DIBR)'], marker='D', label='SCA + EAYB')
plt.xlabel('DIBR Constraints', fontsize=16)
plt.ylabel('Total System Cost', fontsize=16)
plt.title('DIBR Constraints vs. Total System Cost', fontsize=16)
plt.xticks(my_df['DIBR constraints'], fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_cost.png')
plt.clf()

# Plot 8: DIBR Constraints vs. Overall Cache Hit Rate
plt.plot(my_df['DIBR constraints'], my_df['Overall cache hit rate (DIBR)'], marker='o', label='My Algo')
plt.plot(mobile_df['DIBR constraints'], mobile_df['Overall cache hit rate (DIBR)'], marker='s', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['DIBR constraints'], RFP_df['Overall cache hit rate (DIBR)'], marker='^', label='RFP + EAYB')
plt.plot(SCA_df['DIBR constraints'], SCA_df['Overall cache hit rate (DIBR)'], marker='D', label='SCA + EAYB')
plt.xlabel('DIBR Constraints', fontsize=14)
plt.ylabel('Hit rate', fontsize=14)
plt.title('DIBR Constraints vs. Overall Cache Hit Rate', fontsize=16)
plt.xticks(my_df['DIBR constraints'], fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.ylim(0, 0.6)
# plt.yticks([i/5 for i in range(6)])
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_hit_rate.png')
plt.clf()

# Plot 9: DIBR Constraints vs. DIBR Cost
plt.plot(my_df['DIBR constraints'], my_df['DIBR cost'], marker='o', label='My Algo')
plt.plot(mobile_df['DIBR constraints'], mobile_df['DIBR cost'], marker='s', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['DIBR constraints'], RFP_df['DIBR cost'], marker='^', label='RFP + EAYB')
plt.plot(SCA_df['DIBR constraints'], SCA_df['DIBR cost'], marker='D', label='SCA + EAYB')
plt.xlabel('DIBR Constraints', fontsize=14)
plt.ylabel('DIBR Cost', fontsize=14)
plt.title('DIBR Constraints vs. DIBR Cost', fontsize=16)
plt.xticks(my_df['DIBR constraints'], fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_DIBR_cost.png')
plt.clf()

# Plot 10: DIBR Constraints vs. Network Transmission Cost
plt.plot(my_df['DIBR constraints'], my_df['Network Transmission cost'], marker='o', label='My Algo')
plt.plot(mobile_df['DIBR constraints'], mobile_df['Network Transmission cost'], marker='s', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['DIBR constraints'], RFP_df['Network Transmission cost'], marker='^', label='RFP + EAYB')
plt.plot(SCA_df['DIBR constraints'], SCA_df['Network Transmission cost'], marker='D', label='SCA + EAYB')
plt.xlabel('DIBR Constraints', fontsize=14)
plt.ylabel('Network Transmission Cost', fontsize=14)
plt.title('DIBR Constraints vs. Network Transmission Cost', fontsize=16)
plt.xticks(my_df['DIBR constraints'], fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_network_cost.png')
plt.clf()

# Plot 11: DIBR Constraints vs. ISL Hops
plt.plot(my_df['DIBR constraints'], my_df['ISL hops'], marker='o', label='My Algo')
plt.plot(mobile_df['DIBR constraints'], mobile_df['ISL hops'], marker='s', label='Mobile Proxy (EAYB)')
plt.plot(RFP_df['DIBR constraints'], RFP_df['ISL hops'], marker='^', label='RFP + EAYB')
plt.plot(SCA_df['DIBR constraints'], SCA_df['ISL hops'], marker='D', label='SCA + EAYB')
plt.xlabel('DIBR Constraints', fontsize=14)
plt.ylabel('ISL Hops', fontsize=14)
plt.title('DIBR Constraints vs. ISL Hops', fontsize=16)
plt.xticks(my_df['DIBR constraints'], fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_ISL_hops.png')
plt.clf()

# Column plot for Total System Cost comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(my_df['DIBR constraints']))
width = 0.2 
ax.bar([i - 1.5*width for i in x], my_df['Total system cost'], width, label='AMSCC', alpha=0.8)
ax.bar([i - 0.5*width for i in x], mobile_df['Total system cost'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([i + 0.5*width for i in x], RFP_df['Total system cost'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([i + 1.5*width for i in x], SCA_df['Total system cost'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('DIBR Constraints', fontsize=22)
ax.set_ylabel('Total System Cost', fontsize=22)
ax.set_title('DIBR Constraints vs. Total System Cost', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(my_df['DIBR constraints'], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 5000000)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_cost_column.png')
plt.clf()

# Column plot for Hit Rate comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(my_df['DIBR constraints']))
width = 0.2

ax.bar([i - 1.5*width for i in x], my_df['Overall cache hit rate (DIBR)'], width, label='AMSCC', alpha=0.8)
ax.bar([i - 0.5*width for i in x], mobile_df['Overall cache hit rate (DIBR)'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([i + 0.5*width for i in x], RFP_df['Overall cache hit rate (DIBR)'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([i + 1.5*width for i in x], SCA_df['Overall cache hit rate (DIBR)'], width, label='SCA + EAYB', alpha=0.8)

ax.set_xlabel('DIBR Constraints', fontsize=22)
ax.set_ylabel('Hit Rate', fontsize=22)
ax.set_title('DIBR Constraints vs. Overall Cache Hit Rate', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(my_df['DIBR constraints'], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 0.5)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_hit_rate_column.png')
plt.clf()

# Column plot for Network Transmission Cost comparison
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar([i - 1.5*width for i in x], my_df['Network Transmission cost'], width, label='AMSCC', alpha=0.8)
ax.bar([i - 0.5*width for i in x], mobile_df['Network Transmission cost'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([i + 0.5*width for i in x], RFP_df['Network Transmission cost'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([i + 1.5*width for i in x], SCA_df['Network Transmission cost'], width, label='SCA + EAYB', alpha=0.8)

ax.set_xlabel('DIBR Constraints', fontsize=14)
ax.set_ylabel('Network Transmission Cost', fontsize=14)
ax.set_title('DIBR Constraints vs. Network Transmission Cost', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(my_df['DIBR constraints'], fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(1000000, 4500000)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_network_cost_column.png')
plt.clf()

# Column plot for ISL Hops comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([i - 1.5*width for i in x], my_df['ISL hops'], width, label='AMSCC', alpha=0.8)
ax.bar([i - 0.5*width for i in x], mobile_df['ISL hops'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([i + 0.5*width for i in x], RFP_df['ISL hops'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([i + 1.5*width for i in x], SCA_df['ISL hops'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('DIBR Constraints', fontsize=22)
ax.set_ylabel('ISL Hops', fontsize=22)
ax.set_title('DIBR Constraints vs. ISL Hops', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(my_df['DIBR constraints'], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 7)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_ISL_hops_column.png')
plt.clf()

# Column plot for Average Latency comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([i - 1.5*width for i in x], my_df['Average Latency'], width, label='AMSCC', alpha=0.8)
ax.bar([i - 0.5*width for i in x], mobile_df['Average Latency'], width, label='Mobile Proxy (EAYB)', alpha=0.8)
ax.bar([i + 0.5*width for i in x], RFP_df['Average Latency'], width, label='RFP + EAYB', alpha=0.8)
ax.bar([i + 1.5*width for i in x], SCA_df['Average Latency'], width, label='SCA + EAYB', alpha=0.8)
ax.set_xlabel('DIBR Constraints', fontsize=22)
ax.set_ylabel('Average Latency (s)', fontsize=22)
ax.set_title('DIBR Constraints vs. Average Latency', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(my_df['DIBR constraints'], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0.85, 0.97)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation plot/DIBR_constraints_vs_average_latency_column.png')
plt.clf()

print("Finished plotting simulation results.")
