
import speedtest

from robothub_sdk import (
    App,
    Config,
)

class HelloWorld(App):

    def on_configuration(self, old_configuration: Config):
        servers = []
        # If you want to test against a specific server
        # servers = [1234]

        threads = None
        # If you want to use a single threaded test
        # threads = 1

        s = speedtest.Speedtest()
        print("------------------------------------")
        print("Speedtest testing app")
        print("------------------------------------\n")
        print("1. Downloading speedtest server list")
        s.get_servers(servers)
        print("2. Choosing closest server")
        s.get_best_server()
        print("3. Running download speedtest")
        s.download(threads=threads)
        print("4. Running upload speedtest")
        s.upload(threads=threads)
        #s.results.share()
        print("\n\nResults:")
        print("------------------------------------")
        results_dict = s.results.dict()
        print("Download speed: %.1f mbps" % (round(results_dict['download']/(1024*1024),1),))
        print("Upload speed: %.1f mbps" % (round(results_dict['upload']/(1024*1024),1),))
        print("Latency: %i ms" % (round(results_dict['ping'],0),))
        print("Server location: %s, %s" % (results_dict['server']['name'],results_dict['server']['country']))
        print("\n")

app = HelloWorld()
app.run()