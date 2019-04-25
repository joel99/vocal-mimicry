import os
import shutil

def condense_people(uncondensed_root_dir, pdense_root_dir,
                    file_extension, max_people, starting_pid,
                    max_sid):
    last_used_pid = starting_pid

    for pid_offset in range(max_people):
        pid = starting_pid + pid_offset
        if os.path.exists(uncondensed_root_dir + "p" + str(pid)):
            # shutil.copytree(uncondensed_root_dir + "p" + str(pid),
            #                 pdense_root_dir + "p" + str(last_used_pid + 1))
            os.makedirs(pdense_root_dir + "p" + str(last_used_pid),
                        exist_ok=True)
            for sid in range(1, max_sid + 1):
                try:
                    shutil.copy(uncondensed_root_dir + "p" + str(pid)
                                + "/p" + str(pid) + "_{:03d}".format(sid)
                                + file_extension,
                                pdense_root_dir + "p" + str(last_used_pid)
                                + "/p" + str(last_used_pid) + "_{:03d}".format(sid)
                                + file_extension
                    )
                except FileNotFoundError:
                    pass
            last_used_pid += 1
        else:
            print("Person: ", pid, " is missing ")

    print(last_used_pid - starting_pid, " people actually existed in dataset")

    return last_used_pid - starting_pid

def get_shared_sentences(max_people,
                       uncondensed_root_dir,
                       output_root_dir,
                       file_extension,
                       max_sentences=450,
                       starting_pid=1, starting_sid=1,):

    last_used_sid = 0

    for sid_off in range(max_sentences):
        sid = starting_sid + sid_off
        exists_in_all = True
        for pid_offset in range(max_people):
            pid = starting_pid + pid_offset
            if os.path.exists(uncondensed_root_dir + "p" + str(pid)
                              + "/p" + str(pid) + "_{:03d}".format(sid)
                              + file_extension):
                pass
            else:
                print("Person: ", pid, " is missing ", sid)
                exists_in_all = False
                break

        if exists_in_all:
            for pid_offset in range(max_people):
                pid = starting_pid + pid_offset
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

def condense_sentences(max_people,
                       uncondensed_root_dir,
                       output_root_dir,
                       file_extension,
                       max_sentences,
                       starting_pid, starting_sid,):

    min_sentences = 100000000

    for pid_off in range(max_people):
        last_used_sid = starting_sid
        pid = starting_pid + pid_off

        for sid_off in range(max_sentences):
            sid = starting_sid + sid_off
            if os.path.exists(uncondensed_root_dir + "p" + str(pid)
                            + "/p" + str(pid) + "_{:03d}".format(sid)
                            + file_extension):

                os.makedirs(output_root_dir + "p" + str(pid),
                            exist_ok=True)
                shutil.copy(
                    uncondensed_root_dir + "p" + str(pid)
                    + "/p" + str(pid) + "_{:03d}".format(sid)
                    + file_extension,

                    output_root_dir + "p" + str(pid)
                    + "/p" + str(pid) + "_{:03d}".format(last_used_sid)
                    + file_extension
                )
                last_used_sid += 1

        min_sentences = min(min_sentences, last_used_sid)

    print("The minimum amount of sentences anyone has is: ", min_sentences)
    return min_sentences


if __name__ == "__main__":

    ###############################################
    # The following parameters need to be changed #
    ###############################################

    # TODO [Arda] Change the following four params to whatever they should be
    # (in fact pdense_root_dir should probably be under /tmp)
    raw_root_dir = "data/taco/"
    pdense_root_dir = "data/dense_people_mels/"
    output_root_dir = "data/dense_mels/"
    file_extension = ".pt"


    # Shouldnt need to change any of the following
    max_people = 151
    max_sentences = 450
    starting_pid = 225

    n_p = condense_people(uncondensed_root_dir=raw_root_dir,
                          pdense_root_dir=pdense_root_dir,
                          file_extension=file_extension,
                          max_people=151,
                          starting_pid=starting_pid,
                          max_sid=max_sentences)
    # get_shared_sentences(max_people=actual_num_people,
    #                    uncondensed_root_dir=pdense_root_dir,
    #                    output_root_dir=output_root_dir,
    #                    file_extension=file_extension,
    #                    max_sentences=450,
    #                    starting_pid=225, starting_sid=1)
    n_s = condense_sentences(max_people=n_p,
                             uncondensed_root_dir=pdense_root_dir,
                             output_root_dir=output_root_dir,
                             file_extension=file_extension,
                             max_sentences=450,
                             starting_pid=225, starting_sid=1)

    print(n_p, "people each have ", n_s, "sentences each")
