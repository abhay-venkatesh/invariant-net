from lib.ThetaDFSegNet import ThetaDFSegNet

def main():
    dataset_directory = './datasets/Unreal-20View-11class/view0/'
    net = ThetaDFSegNet(dataset_directory)
    net.train(num_iterations=10000, theta=0, is_trainable=True, 
              dataset_directory=dataset_directory, learning_rate=1e-2, batch_size=5)

    # net.test(theta=0, is_trainable=True, dataset_directory=dataset_directory, learning_rate=1e-2)

if __name__ == "__main__":
    main()
