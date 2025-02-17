import argparse

from ga import GA

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ga_med_sur")

    # Setting the parameters for the GA-NAS algorithm
    parser.add_argument('--population_size', type=int, default=5, help='Population Size')
    parser.add_argument('--number_of_generations', type=int, default=5, help='Number of Generations')
    parser.add_argument('--crossover_prob', type=str, default=0.7, help='Crossover Probability')
    parser.add_argument('--mutation_prob', type=int, default=0.6, help='Mutation Probability')
    parser.add_argument('--blocks_size', type=str, default=32, help='Block Size')
    parser.add_argument('--num_classes', type=str, default=10, help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=3, help='Input Channels')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--layers', type=str, default=8, help='Layers')
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

    # Defining the class of GA
    ga = GA(args.population_size, args.number_of_generations, args.crossover_prob, args.mutation_prob, args.blocks_size,
            args.num_classes, args.in_channels, args.epochs, args.batch_size, args.layers, args.n_channels,
            args.dropout_rate, args.retrain, args.resume_train, args.cutout, args.multigpu_num, args.grad_clip,
            args.type_crossover)
    # Running the algorithm
    ga.evolve()
    # indv = [0.76, 0, 0.08, 0, 0.69, 0, 0.15, 1, 0.57, 0, 0.62, 1, 0.1, 2, 0.45, 3, 0.11, 0, 0.42, 0, 0.58, 1, 0.45, 1, 0.75, 0, 0.37, 2, 0.38, 1, 0.5, 1]
    # ga.evaluate_single_model(indv)
    # ga.reload_training(indv)
