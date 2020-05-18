#! /bin/bash
Name=('abc','cde')
Save_Epoch=(2,3)

for i in ${Name[@]}
do
	mkdir ${Name[$i]}
	n=${Name[$i]}
	for e in ${Save_Epoch[@]}
	do
		ep=${Save_epoch[$e]}
		python3 tt.py --name $n --save_epoch $ep | $n$/$ep$.txt

	done

done
