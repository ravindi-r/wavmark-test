# import pdb

import torch
import numpy as np
import tqdm
import time


def decode_trunck(trunck, model, device):
    with torch.no_grad():
        signal = torch.FloatTensor(trunck).to(device).unsqueeze(0)
        message = (model.decode(signal) >= 0.5).int()
        message = message.detach().cpu().numpy().squeeze()
    return message


def extract_watermark_v3_batch(data, start_bit, shift_range, num_point, model, device, batch_size=10,
                               shift_range_p=0.5, show_progress=False):
    #print("\textract_watermark_v3_batch start: ", data.shape)
    print("data: ", data.shape)
    assert type(show_progress) == bool
    start_time = time.time()
    # 1.determine the shift step length:
    shift_step = int(shift_range * num_point * shift_range_p)
    print("shift_step: ", shift_step)

    # 2.determine where to perform detection
    # pdb.set_trace()
    total_detections = (len(data) - num_point) // shift_step
    print("total_detections: ", total_detections)
    total_detect_points = [i * shift_step for i in range(total_detections)]

    # 3.construct batch for detection
    total_batch_counts = len(total_detect_points) // batch_size + 1
    print("total_batch_counts: ", total_batch_counts)
    results = []

    # ravi
    output_chunks = []

    the_iter = range(total_batch_counts)
    if show_progress:
        the_iter = tqdm.tqdm(range(total_batch_counts))

    for i in the_iter:

        print("start")
        detect_points = total_detect_points[i * batch_size:i * batch_size + batch_size]
        print("\tdetect_points: ", detect_points)

        if len(detect_points) == 0:
            break

        current_batch = np.array([data[p:p + num_point] for p in detect_points])
        print("\tcurrent_batch: ", current_batch.shape)

        with torch.no_grad():
            signal = torch.FloatTensor(current_batch).to(device)
            print("\tsignal: ", signal.shape)
            
            var = model.decode(signal)[1]
            
            # ravi
            signal_restored_1 = model.decode(signal)[0]
            print("\tsignal_restored_1: ", signal_restored_1.shape)

            signal_restored = signal_restored_1.detach().cpu().numpy().squeeze()
            print("\tsignal_restored_2: ", signal_restored.shape)

            # ravi
            assert signal_restored.shape == signal.shape
            output_chunks.append(signal_restored)

            # batch_message = (model.decode(signal) >= 0.5).int().detach().cpu().numpy()

            batch_message = (var >= 0.5).int().detach().cpu().numpy()

            for p, bit_array in zip(detect_points, batch_message):
                decoded_start_bit = bit_array[0:len(start_bit)]
                ber_start_bit = 1 - np.mean(start_bit == decoded_start_bit)
                num_equal_bits = np.sum(start_bit == decoded_start_bit)
                if ber_start_bit > 0:  # exact match
                    continue
                results.append({
                    "sim": 1 - ber_start_bit,
                    "num_equal_bits": num_equal_bits,
                    "msg": bit_array,
                    "start_position": p,
                    "start_time_position": p / 16000
                })

        print("end")

    # ravi
    reconstructed_array = np.concatenate(output_chunks)
    print("reconstructed_array: ", reconstructed_array.shape)
    
    #for chunk in output_chunks:
    #    print("chunk shape: ", chunk.shape)
                

    end_time = time.time()
    time_cost = end_time - start_time

    info = {
        "time_cost": time_cost,
        "results": results,
    }

    if len(results) == 0:
        return None, info

    results_1 = [i["msg"] for i in results if np.isclose(i["sim"], 1.0)]
    mean_result = (np.array(results_1).mean(axis=0) >= 0.5).astype(int)



    # ravi
    # signal_2 = torch.FloatTensor(data).to(device)
    # signal_restored_2 = model.decode(signal_2)[0]
    # assert signal_restored_2.shape == data.shape
    # print("signal_restored_2: ", signal_restored_2.shape)
    # output_chunks.append(signal_restored)
    


    #print("\textract_watermark_v3_batch start: ", reconstructed_array.shape)
    # return mean_result, info, signal_restored_2
    return mean_result, info, reconstructed_array
