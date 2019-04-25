import os
import shutil

###############################################
# The following parameters need to be changed #
###############################################
# TODO Arda update to what you're using
# Make sure thse directories have the trailing slash and are specified
# relative to the working directory (ie from where script is run)

uncondensed_root_dir = "data/mel/"
output_root_dir = "data/condensed_mel/"
file_extension = ".npy"
###############################################

#######################################################################
# Try reducing max_people if not enough data remains after condensing #
#######################################################################

max_people = 151
max_sentences = 450

#######################################################################
# Beyond bugfixes, shouldn't need to touch anything beyond this point #
#######################################################################

starting_pid = 225
starting_sid = 1

if __name__ == "__main__":

    last_used_sid = 0

    for sid_off in range(max_sentences):
        sid = starting_sid + check_sid
        exists_in_all = True
        for pid_offset in range(num_people):
            pid = starting_pid + pid_offset
            if os.path.exists(uncondensed_root_dir + "p" + str(pid)
                              + "/p" + str(pid) + "_{:03d}".format(sid)
                              + file_extension):
                pass
            else:
                exists_in_all = False
                break

        if exists_in_all:
            for pid_offset in range(num_people):
                os.makedirs(output_root_dir + "p" + str(pid),
                            exist_ok=True)
                shutil.copy(
                    uncondensed_root_dir + "p" + str(pid)
                    + "/p" + str(pid) + "_{:03d}".format(sid)
                    + file_extension,

                    output_root_dir + "p" + str(pid)
                    + "/p" + str(pid) + "_{:03d}".format(last_used_sid + 1)
                    + file_extension
                )
            last_used_sid += 1

    print(last_used_sid, " sentences were shared across ",
          max_people, " people")
