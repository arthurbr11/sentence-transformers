#Step 1: Get the list of nodes and states for "hopper-ex" jobs
squeue -h -o "%.10P %.10u %.2t %.15R %.10i" | grep "hopper-ex" | awk '{print $2, $3, $4, $5}' > job_nodes.txt

# Step 2: Get the GPU count for each job
scontrol show job | awk '
/^JobId=/ {
    job=$1;
        gsub(/JobId=/, "", job);
            base_job=job;
                sub(/_[0-9]+$/, "", base_job);  # Strip array index if present
        }
/TRES=/ {
    match($0, /gres\/gpu=([0-9]+)/, arr);
        if (arr[1] != "") {
                        job_gpus[base_job]=arr[1];
                            }
            }
    END {
        for (job in job_gpus) print job, job_gpus[job];
        }' > job_gpus.txt

# Step 3: Calculate running and pending nodes and GPUs
squeue -h -o "%.10i %.10P %.10j %.10u %.2t %.10M %.6D %.6b %.15R" | grep "hopper-ex" | awk '
{
            job_id = $1;
                base_job_id = job_id;
                    sub(/_[0-9]+$/, "", base_job_id);  # Strip array index if present
                        if ($5 == "R") {
                                        running_nodes[$4] += $7;
                                                running_jobs[$4][base_job_id] = $7;  # Save base job id and associated nodes
                                                    } else if ($5 == "PD") {
                                                            pending_nodes[$4] += $7;
                                                                    pending_jobs[$5][base_job_id] = $7;  # Save base job id and associated nodes
                                                                        }
                                                        }
                                                END {
                                                    for (user in running_nodes) {
                                                                    printf "%-10s %-13d %-12d %-12d %-12d\n", user, running_nodes[user], 0, (pending_nodes[user] ? pending_nodes[user] : 0), 0;
                                                                        }
                                                                    for (user in pending_nodes) {
                                                                                    if (!(user in running_nodes)) {
                                                                                                        printf "%-10s %-13d %-12d %-12d %-12d\n", user, 0, 0, pending_nodes[user], 0;
                                                                                                                }
                                                                                                            }
                                                                                            }' > temp_summary.txt

                                                                                    # Step 4: Get total number of nodes in the cluster
                                                                                    total_nodes=$(sinfo -N -h -p hopper-extra | wc -l)

                                                                                    # Combine the GPU counts and print the final summary, sorted by running GPUs
                                                                                    awk -v total_nodes="$total_nodes" '
                                                                                    BEGIN {
                                                                                        print "User       Running_Nodes Running_GPUs Pending_Nodes Pending_GPUs";
                                                                                }
                                                                        NR==FNR {job_gpus[$1]=$2; next} 
                                                                        {
                                                                                    user=$1; state=$2; job=$4;
                                                                                        base_job = job;
                                                                                            sub(/_[0-9]+$/, "", base_job);  # Strip array index if present
                                                                                                gpus = (base_job in job_gpus) ? job_gpus[base_job] : 0;
                                                                                                    if (state == "R") {
                                                                                                                    running_gpus[user] += gpus;
                                                                                                                        } else if (state == "PD") {
                                                                                                                                pending_gpus[user] += gpus;
                                                                                                                                    }
                                                                                                                    }
                                                                                                            END {
                                                                                                                while ((getline line < "temp_summary.txt") > 0) {
                                                                                                                                split(line, fields);
                                                                                                                                        user = fields[1];
                                                                                                                                                running_gpus_count = (running_gpus[user] ? running_gpus[user] : 0);
                                                                                                                                                        pending_gpus_count = (pending_gpus[user] ? pending_gpus[user] : 0);
                                                                                                                                                                if (fields[1] != "User") {
                                                                                                                                                                                    summary[user] = sprintf("%-10s %-13d %-12d %-12d %-12d\n", fields[1], fields[2], running_gpus_count, fields[4], pending_gpus_count);
                                                                                                                                                                                                running_gpus_sorted[user] = running_gpus_count;
                                                                                                                                                                                                        }
                                                                                                                                                                                                    }
                                                                                                                                                                                                # Sort by running GPUs
                                                                                                                                                                                                    n = asorti(running_gpus_sorted, sorted_users, "@val_num_desc")
                                                                                                                                                                                                        for (i = 1; i <= n; i++) {
                                                                                                                                                                                                                        user = sorted_users[i];
                                                                                                                                                                                                                                printf "%s", summary[user];
                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                print "----------------------------------------------";
                                                                                                                                                                                                                                    print "Total Nodes Available: " total_nodes;
                                                                                                                                                                                                                            }' job_gpus.txt job_nodes.txt

                                                                                                                                                                                                                    # Cleanup: Remove temporary files
                                                                                                                                                                                                                    rm -f job_nodes.txt job_gpus.txt temp_summary.txt
