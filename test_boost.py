import mmap
import os
import struct

def read_shared_memory(shm_name, shm_size):
    """
    Read data from a Boost Shared Memory space.

    Args:
    shm_name (str): Name of the shared memory space.
    shm_size (int): Size of the shared memory space.

    Returns:
    bytes: Data read from the shared memory space.
    """
    try:
        # Open the shared memory file
        fd = os.open(shm_name, os.O_RDONLY)
        
        # Memory map the shared memory file
        mmapped_file = mmap.mmap(fd, shm_size, mmap.MAP_SHARED, mmap.PROT_READ)
        
        # Read data from the mapped memory
        data = mmapped_file.read(shm_size)
        
        # Close the file descriptor and unmap the memory
        mmapped_file.close()
        os.close(fd)
        
        return data
    except Exception as e:
        print(f"Error reading shared memory: {e}")
        return None

if __name__ == "__main__":
    # Shared memory parameters
    shm_name = "myshm"  # Path to the shared memory file
    shm_size = 1024  # Size of the shared memory space in bytes
    
    # Read from shared memory
    data = read_shared_memory(shm_name, shm_size)
    
    if data is not None:
        # Assuming the data in shared memory is a string
        print(f"Read data from shared memory: {data.decode('utf-8')}")
    else:
        print("Failed to read from shared memory")
