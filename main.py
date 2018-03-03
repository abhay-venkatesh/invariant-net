# Only use a single GPU when not testing
if os.name != 'nt': 
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from lib.SegNetLogger import SegNetLogger 

def main():
    dataset_directory = './datasets/Unreal-20View-11class/'
    net = SegNetLogger()
    net.train(num_iterations=500, dataset_directory=dataset_directory, learning_rate=1e-2, batch_size=5)
    # net.build_contextual_feedback_log(dataset_directory)

if __name__ == "__main__":
    main()
