#-*-encoding:utf-8-*-#

def divide_txt(file_name, size):
    fp = open(file_name, "r")
    part_cnt = 0
    with open(file_name.replace(".txt", "_part" + str(part_cnt) + ".txt"), "w") as temp:
        size_cnt = 0
        buf = fp.readline()
        while buf:
            temp.write(buf + "\n")
            buf= fp.readline()
            size_cnt += 1
            if(size_cnt == size):
                print "file part %d end"%(part_cnt)
                part_cnt += 1
                break
    print "program end"
    fp.close()

if __name__=="__main__":
    dir = "D:\广告2"
    file = "./tag_did_media_bundle.20180918.txt"
    divide_txt(file, 10240)
