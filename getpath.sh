#! /bin/bash
a=`expr 1`
cd '20230702'
# dir='./20230702/1lux_infrared_RGB/'
# dir='./20230702/3lux_infrared_RGB/'
# dir='./20230702/5lux_corridor_RGB/'
# dir='./20230702/300lux_office_RGB/'
# dir='./20230702/400lux_office_RGB/'
# dir='./20230702/output_1lux_infrared_RGB_3/'
# dir='output_3lux_infrared_RGB_3'
# dir='output_5lux_corridor_RGB_3'
# dir='output_300lux_office_RGB_3'
dir='output_400lux_office_RGB_3'

cd $dir

# for image in ./*.jpg
for image in ./*.png
    do
    a=`expr $a + 1`
    b=`expr $a % 1`
    if [ $b == 0 ]
    then
        realpath $image >> $dir'.txt'
    fi
done
mv $dir'.txt' /home/chenxiang/code/d2net_simpletracker