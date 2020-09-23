# import sys
# import subprocess
#
# procs = []
# proc1 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=0', 'python', 'multi_run.py', 'multi_results/multi_config_12.json'])
# procs.append(proc1)
# proc2 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=1', 'python', 'multi_run.py', 'multi_results/multi_config_123.json'])
# procs.append(proc2)
# proc3 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=2', 'python', 'multi_run.py', 'multi_results/multi_config_1234.json'])
# procs.append(proc3)
# proc4 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=3', 'python', 'multi_run.py', 'multi_results/multi_config_124.json'])
# procs.append(proc4)
# proc5 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=4', 'python', 'multi_run.py', 'multi_results/multi_config_13.json'])
# procs.append(proc5)
# proc6 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=5', 'python', 'multi_run.py', 'multi_results/multi_config_134.json'])
# procs.append(proc6)
# proc7 = subprocess.Popen(['CUDA_VISIBLE_DEVICES=6', 'python', 'multi_run.py', 'multi_results/multi_config_14.json'])
# procs.append(proc7)
#
# for proc in procs:
#     proc.wait()


# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results/multi_config_1.json      # 495
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results/multi_config_12.json      # 497
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results/multi_config_123.json      # 498
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results/multi_config_1234.json      # 499
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results/multi_config_124.json      # 500
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results/multi_config_13.json      # 501
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results/multi_config_134.json      # 502
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results/multi_config_14.json      # 503

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_8_4/multi_config_12.json      # 504
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_8_4/multi_config_123.json      # 505
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_8_4/multi_config_1234.json      # 506
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_8_4/multi_config_124.json      # 507
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_8_4/multi_config_13.json      # 508
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_8_4/multi_config_134.json      # 509
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_8_4/multi_config_14.json      # 510
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_8_4/multi_config_1.json      # 511

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_8_16/multi_config_12.json      # 512
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_8_16/multi_config_123.json      # 513
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_8_16/multi_config_1234.json      # 514
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_8_16/multi_config_124.json      # 515
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_8_16/multi_config_13.json      # 516
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_8_16/multi_config_134.json      # 517
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_8_16/multi_config_14.json      # 518
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_8_16/multi_config_1.json      # 519

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_8_2/multi_config_12.json      # 520
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_8_2/multi_config_123.json      # 521
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_8_2/multi_config_1234.json      # 522
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_8_2/multi_config_124.json      # 523
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_8_2/multi_config_13.json      # 524
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_8_2/multi_config_134.json      # 525
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_8_2/multi_config_14.json      # 526
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_8_2/multi_config_1.json      # 527



# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_1e-5/multi_config_1234.json      # 528
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_2e-5/multi_config_1234.json      # 529
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_5e-5/multi_config_1234.json      # 530
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_1e-5/multi_config_124.json      # 531
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_2e-5/multi_config_124.json      # 532
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_5e-5/multi_config_124.json      # 533
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1e-5/multi_config_134.json      # 534
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2e-5/multi_config_134.json      # 535

# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_5e-5/multi_config_134.json      # 536
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2e-5/multi_config_14.json      # 537
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_5e-5/multi_config_14.json      # 538
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1e-5/multi_config_123.json      # 539
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2e-5/multi_config_123.json      # 540
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_5e-5/multi_config_123.json      # 541
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1e-5/multi_config_12.json      # 542
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_1e-5/multi_config_14.json      # 543

# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_2e-5/multi_config_12.json      # 544
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_5e-5/multi_config_12.json      # 545
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_1e-5/multi_config_13.json      # 546
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_2e-5/multi_config_13.json      # 547

# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_5e-5/multi_config_13.json      # 548

# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_1e-5/multi_config_1.json      # 549
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2e-5/multi_config_1.json      # 550
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_5e-5/multi_config_1.json      # 551

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_1e-5/multi_config_1_sent.json      # 552
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2e-5/multi_config_12_sent.json      # 553

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_1e-5/multi_config_1234.json      # 556
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_2e-5/multi_config_1234.json      # 557
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_5e-5/multi_config_1234.json      # 558
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_1e-5/multi_config_124.json      # 559
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_2e-5/multi_config_124.json      # 560   <1
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_5e-5/multi_config_124.json      # 561
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_1e-5/multi_config_134.json      # 562   <2
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_2e-5/multi_config_134.json      # 563

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_5e-5/multi_config_134.json      # 564
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_2e-5/multi_config_14.json      # 565
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_5e-5/multi_config_14.json      # 566
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_1e-5/multi_config_123.json      # 567
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_2e-5/multi_config_123.json      # 568
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_5e-5/multi_config_123.json      # 569
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_1e-5/multi_config_12.json      # 570
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1e-5/multi_config_14.json      # 571

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2e-5/multi_config_12.json      # 572
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_5e-5/multi_config_12.json      # 573
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1e-5/multi_config_13.json      # 574
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_2e-5/multi_config_13.json      # 575
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_5e-5/multi_config_13.json      # 576
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_1e-5/multi_config_1.json      # 577
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_2e-5/multi_config_1.json      # 578
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_5e-5/multi_config_1.json      # 579

# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_2e-5/multi_config_124_alt.json      # 582   <1
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_1e-5/multi_config_134_alt.json      # 583   <2

"""
MULTI 4
"""

# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_1/multi_config_1234.json      # 588  <
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2/multi_config_1234.json      # 589  <
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_3/multi_config_1234.json      # 590  <

# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1/multi_config_124.json      # 591   <
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_2/multi_config_124.json      # 592   <
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_3/multi_config_124.json      # 593   <

# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_1/multi_config_134.json      # 594   <
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_2/multi_config_134.json      # 595   <
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_3/multi_config_134.json      # 596   <

# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_1/multi_config_14.json      # 597    <
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_2/multi_config_14.json      # 598    <
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_3/multi_config_14.json      # 599    <


# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_1/multi_config_123.json      # 600
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_2/multi_config_123.json      # 601
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_3/multi_config_123.json      # 602

# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_1/multi_config_12.json      # 603
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2/multi_config_12.json      # 604
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_3/multi_config_12.json      # 605

# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1/multi_config_13.json      # 606
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2/multi_config_13.json      # 607
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_3/multi_config_13.json      # 608

# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1/multi_config_1.json      # 609
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_2/multi_config_1.json      # 610
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_3/multi_config_1.json      # 611


################# RERUN MATRES ###################

# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_1/multi_config_1234.json      #
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2/multi_config_1234.json      #
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_3/multi_config_1234.json      #

# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1/multi_config_124.json      #
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_2/multi_config_124.json      #
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_3/multi_config_124.json      #

# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_1/multi_config_134.json      #
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2/multi_config_134.json      #
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_3/multi_config_134.json      #

# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1/multi_config_14.json      #
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_2/multi_config_14.json      #
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_3/multi_config_14.json      #

################# TRY DURATION (TRAIN vs ALL) ###################

# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_1/multi_config_15.json      #
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_2/multi_config_15.json      #
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_3/multi_config_15.json      #

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_1/multi_config_15.json      #
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_2/multi_config_15.json      #
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_3/multi_config_15.json      #

################# WITH DURATION ###################

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_1/multi_config_12345.json      #
# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2/multi_config_12345.json      #
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_3/multi_config_12345.json      #

# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1/multi_config_1245.json      #
# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2/multi_config_1245.json      #
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_3/multi_config_1245.json      #

# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_2/multi_config_1345.json      #
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_3/multi_config_1345.json      #

# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_1/multi_config_145.json      #
# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1/multi_config_1345.json      #

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2/multi_config_145.json      #
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_3/multi_config_145.json      #


# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1/multi_config_1235.json      #

# CUDA_VISIBLE_DEVICES=0 python multi_run.py multi_results_2/multi_config_1235.json      #
# CUDA_VISIBLE_DEVICES=1 python multi_run.py multi_results_3/multi_config_1235.json      #

# CUDA_VISIBLE_DEVICES=2 python multi_run.py multi_results_1/multi_config_125.json      #
# CUDA_VISIBLE_DEVICES=3 python multi_run.py multi_results_2/multi_config_125.json      #
# CUDA_VISIBLE_DEVICES=4 python multi_run.py multi_results_3/multi_config_125.json      #

# CUDA_VISIBLE_DEVICES=5 python multi_run.py multi_results_1/multi_config_135.json      #
# CUDA_VISIBLE_DEVICES=6 python multi_run.py multi_results_2/multi_config_135.json      #
# CUDA_VISIBLE_DEVICES=7 python multi_run.py multi_results_3/multi_config_135.json      #