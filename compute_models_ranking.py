import os

def print_results(results, final_iterations):
    for n_clusters, d in results.items():
        print("\n\n------------------------------------")
        print("CLUSTERS: "+str(n_clusters))
        print("------------------------------------")
        for category_name, category in d.items():
            print("\nFitness function: "+category_name)
            print("------------------------------------")
            print("Model name\tValue\tIteration")
            print("------------------------------------")
            for model_name, result in category.items():
                if result == 0:
                    print(model_name+"\t\tNo improvement\t\t-")
                else:
                    print(model_name+"\t\t"+str(result)+"\t\t"+str(final_iterations[model_name]))


def models_ranking(snapshots_dir, early_stopping_patience = 10):
    results = {
                2: {"CentroidDistance" : {}, "MinimumPointDistance" : {}},
                4: {"CentroidDistance" : {}, "MinimumPointDistance" : {}}
              }
    final_iteration = {}
    for filename in os.listdir(snapshots_dir):
        if filename.endswith("normalized"):
            continue
        
        if filename.find("2_clusters")>0:
            n_clusters = 2
        else:
            n_clusters = 4

        if filename.endswith("CentroidDistance"):
            category = "CentroidDistance"
        else:
            category = "MinimumPointDistance"

        scores_file = os.path.join(snapshots_dir, filename, "model.scores")
        if not os.path.exists(scores_file):
            results[n_clusters][category][filename] = 0
            final_iteration[filename] = -1
        else:
            lines = []
            with open(scores_file, mode = "r") as f:
                lines = f.readlines()
            results[n_clusters][category][filename] = float(lines[-1])
            iteration = int(len(lines)/4 - early_stopping_patience - 2)
            final_iteration[filename] = 0 if iteration<0 else iteration

    for n_clusters, d in results.items():
        for category_name, category in d.items():
            results[n_clusters][category_name] = {k: results[n_clusters][category_name][k] for k in sorted(results[n_clusters][category_name], key=results[n_clusters][category_name].get, reverse=True)}
    
    print_results(results, final_iteration)

if __name__=="__main__":
    models_ranking("snapshots_")