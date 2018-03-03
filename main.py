from lib.SegNetLogger import SegNetLogger 

def main():
    dataset_directory = './datasets/UnrealFlows/'
    net = UnfixedMultiSegNet(dataset_directory)
    net.train(num_iterations=100000, theta=0, is_trainable=True, 
              dataset_directory=dataset_directory, learning_rate=1e-2, batch_size=5)

if __name__ == "__main__":
    main()
