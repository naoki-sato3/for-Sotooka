# for-Sotooka

PyTorchのTutorialsはかなり参考になるから知っておいた方がいいですね。まずは[基礎](https://pytorch.org/tutorials/beginner/basics/intro.html)から。
これの順序に沿って読んでくだけで基礎は分かります。  
タスクごとにもTutorialsはあるから、それも知っておいた方がいいです。
例えば、[画像分類](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)とか、[GAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)とかね。  

変なサイトで勉強するよりも公式の説明の方がいいです。PyTorchのoptimizerを司ってる[torch.optim](https://pytorch.org/docs/stable/optim.html)とか。  
例えば、Adamの引数はどんななのかなって思ったら、[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)のページを見に行きます。  
<img width="821" alt="スクリーンショット 2023-02-24 15 19 46" src="https://user-images.githubusercontent.com/95958702/221106736-3b1c149c-e918-46ca-9d30-22f98eb8bd13.png">  
これを見れば、Adamにはデフォルトで学習率0.001が与えられていることとか、AMSGradを使いたかったら、引数amsgradにTrueを渡せばいいこととかが分かります。
他にも、その下にはどんな挙動で点列を更新するように作られてるのかが書いてありますね。

っていう公式の説明と、実際にどんな風に書けば使えるのかっていうTutorialsのプログラムを見比べるのが一番速いです。  
hiroya.pyは僕が今使ってるコードのコピーで、CIFAR10（データセットの名前）の画像分類タスク用のプログラムです。色んなoptimizerを使ってるから、指定の仕方の参考になると思います。  
でも、そのままダウンロードしても動かないはずですから、見るだけにしてください。ごめんね。  

hiroya.pyはこの[github](https://github.com/kuangliu/pytorch-cifar)を参考にして作られています。こっちなら、実行の方法も書いてあるからすぐに動かせますよ。

1epochごとに分類精度を計算してグラフにすると、例えばこんな風になります。AMSGradの強さが分かりますね。
<img width="1142" alt="スクリーンショット 2023-02-24 15 52 02" src="https://user-images.githubusercontent.com/95958702/221112016-6045c07d-a76b-4e27-bde3-c98b2198227c.png">
