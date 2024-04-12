
import multiprocessing
import mmap
import os

def create_shared_memory(shm_name, shm_size):
    """
    Create a shared memory segment.

    Args:
    shm_name (str): Name of the shared memory segment.
    shm_size (int): Size of the shared memory segment in bytes.

    Returns:
    multiprocessing.shared_memory.SharedMemory: Shared memory object.
    """
    try:
        # Create a shared memory object
        shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
        return shm
    except Exception as e:
        print(f"Error creating shared memory: {e}")
        return None

def write_to_shared_memory(shm_name, value):
    """
    Write data to a shared memory segment.

    Args:
    shm_name (str): Name of the shared memory segment.
    value: Data to write to the shared memory segment.
    """
    try:
        # Open the existing shared memory segment
        shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)

        # Memory map the shared memory segment
        mmapped_file = mmap.mmap(shm.fd, shm.size)

        # Write data to the mapped memory
        mmapped_file.write(value)

        # Close the memory map and shared memory object
        mmapped_file.close()
        shm.close()
    except Exception as e:
        print(f"Error writing to shared memory: {e}")

if __name__ == "__main__":
    # Shared memory parameters
    shm_name = "MySharedMemory"
    shm_size = 4096  # Size of the shared memory segment in bytes

    # Create shared memory
    shm = create_shared_memory(shm_name, shm_size)

    if shm is not None:
        # Write data to shared memory
        data_to_write = b'10'  # Convert integer 10 to bytes
        write_to_shared_memory(shm_name, data_to_write)
        print(f"Data '10' written to shared memory '{shm_name}'")

        input()
    else:
        print("Failed to create shared memory")
