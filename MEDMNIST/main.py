import argparse
import time

from ga import GA

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ga_med_sur")

    # Setting the parameters for the GA-NAS algorithm
    parser.add_argument('--population_size', type=int, default=2, help='Population Size')
    parser.add_argument('--number_of_generations', type=int, default=30, help='Number of Generations')
    parser.add_argument('--crossover_prob', type=str, default=0.7, help='Crossover Probability')
    parser.add_argument('--mutation_prob', type=int, default=0.6, help='Mutation Probability')
    parser.add_argument('--blocks_size', type=str, default=32, help='Block Size')
    parser.add_argument('--num_classes', type=str, default=10, help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=3, help='Input Channels')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')
    parser.add_argument('--layers', type=str, default=6, help='Layers')
    parser.add_argument('--n_channels', type=str, default=16, help='Number of Channels')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--retrain', type=str, default=False, help='Resume Retraining')
    parser.add_argument('--resume_train', type=str, default=False, help='Resume Training')
    parser.add_argument('--cutout', type=str, default=False, help='Cutout')
    parser.add_argument('--multigpu_num', type=str, default=3, help='Multi GPU Number')
    parser.add_argument('--grad_clip', type=str, default=5)
    parser.add_argument('--type_crossover', type=str, default='one-point',
                        choices=['one-point', 'two-point', 'uniform'])

    args = parser.parse_args()

    # Setting the MedMNIST datasets
    datasets = ['breastmnist']
    #Iterate through the datasets
    for dataset in datasets:
        #Initialize the genetic algorithm based NAS parameters
        ga = GA(args.population_size, args.number_of_generations, args.crossover_prob, args.mutation_prob,
                args.blocks_size,
                args.num_classes, args.in_channels, args.epochs, args.batch_size, args.layers, args.n_channels,
                args.dropout_rate, args.retrain, args.resume_train, args.cutout, args.multigpu_num, args.grad_clip,
                'MEDMNIST', dataset, args.type_crossover)
        # ga = GA(20,30 ,0.9,0.6,32,10,3,3,1024,8,16,0.3,False,False,False,3,5,'MEDMNIST',dataset,type_crossover='one-point')
        # Running the algorithm
        # Start timing
        start_time = time.time()
        #ga.evaluate_single_model( [0.6364104112637804, 0, 0.7555511385430487, 0, 0.808120379564417, 0, 0.5393422419156507, 1, 0.4271077886262563, 0, 0.22210781047073025, 1, 0.7030189588951778, 1, 0.30087830981676966, 1, 0.27864646423661144, 0, 0.2420552715115004, 0, 0.6323058305935795, 0, 0.18651851039985423, 1, 0.22649577519793795, 2, 0.13752094414599325, 2, 0.659984046034179, 2, 0.8972157579533268, 1])
        # Running the algorithm
        ga.evolve()

        # End timing
        end_time = time.time()

        # Compute and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Time taken to run the algorithm: {elapsed_time} seconds")

