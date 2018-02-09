from lib.ThetaDFSegNet import ThetaDFSegNet

def main():
  dataset_directory = './datasets/Unreal-20View-11class/view0'
  net = ThetaDFSegNet(dataset_directory)
  self, num_iterations, theta, is_trainable, dataset_directory,
              learning_rate=0.1, batch_size=5):
  net.train(num_iterations=10000, theta=0, is_trainable=True learning_rate=1e-2)
  # net.test_sequence()

if __name__ == "__main__":
  main()
