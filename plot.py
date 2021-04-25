from got10k.experiments import ExperimentGOT10k

report_files = ['reports/GOT-10k/TLD/performance.json',"reports/GOT-10k/TLD_momentum/performance.json", "reports/GOT-10k/TLD_momentum0.2/performance.json", "reports/GOT-10k/TLD_momentum0.5/performance.json"]
tracker_names = ['TLD', 'TLD_momentum', 'TLD_momentum0.2', 'TLD_momentum0.5']

# setup experiment and plot curves
experiment = ExperimentGOT10k('data/GOT-10k', subset='val')
experiment.plot_curves(report_files, tracker_names)