# You may NOT alter the import list!!!!
import pyshark
import hashlib

class MITMException(Exception):
    """A class to throw if you come across incorrect syntax or other issues"""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MITMProject(object):
    # HINT:
    #  You can use the __init__ method to load the packet capture
    def __init__(self):
        self.cap = pyshark.FileCapture('flag6.pcap')
        self.class_id = "CS6035F24"

        # TODO: Change this to YOUR Georgia Tech ID!!!
        # This is your 9-digit Georgia Tech ID
        self.student_id = '904209565'

    def get_student_hash(self, value):
        return hashlib.sha256(self.student_id.encode('UTF-8') + self.class_id + value).hexdigest()

    # TODO: 
    #   Task 1: Return n being:
    #       n = Number of ICMP Packets
    def icmp_count(self):
        self.cap.reset()
        n = 0
        # TODO: Implement me 
        for packet in self.cap:
            if 'ICMP' in packet:
                n+=1



        return n
    
    # TODO: 
    #   Task 2: Return r,a, being:
    #       r = Number of ICMP Echo Requests
    #       a = ICMP Echo Reply
    def icmp_request_reply(self):
        self.cap.reset()
        icmp_type=""
        r = 0
        a = 0
        # TODO: Implement me 
        for packet in self.cap:
            if 'ICMP' in packet:
                icmp_type = packet.icmp.type
                if icmp_type == '8':
                    r+=1
                if icmp_type == '0':
                    a+=1


        return r,a 

    # TODO: 
    #   Task 3: Return m,n, being:
    #       m = Most Common Destination MAC Address
    #       n = Number of Occurrences
    def dest_mac(self):
        self.cap.reset()
        m,n = 0,0 
      # TODO: Implement me 
        mac_addr = ""
        key = 0
        mac_addresses = {}
        for packet in self.cap:
            if 'ETH' in packet:
                mac_addr = packet.eth.dst
                if mac_addr in mac_addresses:
                    mac_addresses[mac_addr]+=1
                else:
                    mac_addresses[mac_addr]=1


        key = max(mac_addresses, key=mac_addresses.get)
        m = key
        n = mac_addresses[m]

        return m,n


if __name__ == '__main__':
    pcap_analysis = MITMProject()
    icmp_count = pcap_analysis.icmp_count()
    request,reply = pcap_analysis.icmp_request_reply()
    dest_mac,occurences = pcap_analysis.dest_mac()
    print("Number of ICMP Packets  : ", icmp_count)
    print("Number of ICMP Requests and Replies : ",request,reply)
    print("Most Common MAC Address and Number of Ocurrences: ", dest_mac,occurences)

    
