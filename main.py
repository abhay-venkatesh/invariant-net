from lib.InvariantNet import InvariantNet

def main():
    dataset_directory = './datasets/Unreal-20View-11class/view0/'

    net = InvariantNet()
    net.train(dataset_directory, num_iterations=500, 
              learning_rate=1e-2, batch_size=5)

if __name__ == "__main__":
    main()
