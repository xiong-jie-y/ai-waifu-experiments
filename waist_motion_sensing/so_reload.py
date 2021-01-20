from watchdog.observers import Observer
# from watchdog import 

import streamlit as st
import ctypes

# import string_sum as ss
import os
import faulthandler
faulthandler.enable()

import so_wrapper

so_wrapper.run_so_funcs()
# ss = ctypes.PyDLL('string_sum.so')

# st.write(dir(ss.__file__))

# dll = ctypes.CDLL(ss.__file__)
# print("loaded")
# st.write(ss.__file__)
# st.write(ss.rust_fn())
# st.write("32")
# st.write(ss.sum_as_string(1, 2))


# # # handle = ss._handle

# # # while isLoaded('./string_sum.so'):
# #     # dlclose(handle)

# def isLoaded(lib):
#    libp = os.path.abspath(lib)
#    ret = os.system("lsof -p %d | grep %s > /dev/null" % (os.getpid(), libp))
#    return (ret == 0)

# def dlclose(handle):
#    libdl = ctypes.CDLL("libdl.so")
#    libdl.dlclose.argtypes = [ctypes.c_void_p]
#    libdl.dlclose(handle)

# # handle = ss._handle
# # del ss
# if isLoaded('string_sum.so'):
#     # st.write('unloading')
#     dlclose(ss._handle)
#     # break
#     # break
# del ss

# def update_dummy_module():
#     # Rewrite the dummy.py module. Because this script imports dummy,
#     # modifiying dummy.py will cause Streamlit to rerun this script.
#     dummy_path = string_sum.dummy.__file__
    
#     # with open(dummy_path, "w") as fp:
#     #     fp.write(f'timestamp = "{dt.datetime.now()}"')

# @st.cache
# def install_monitor():
#     # watchdog = Watchdog()
#     # watchdog.hook = ... <-- We'll deal with this next
#     observer = Observer()
#     observer.schedule(update_dummy_module, path=".", recursive=False)
#     observer.start()

# install_monitor()
