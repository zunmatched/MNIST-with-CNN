# MNIST-with-CNN
使用tensorflow 2.0中內建的keras透過MNIST資料庫對CNN進行訓練

初次使用tensorflow的CNN架構，如有任何不專業之處歡迎指教。
本次參考tensorflow示範，使用MNIST訓練CNN模型。
與直接使用權連接網路不同，我們雖然不用對資料健行flatten，但是必須將資料的維度提升到4D。
由於此腳本旨在確認MNIST在CNN中是否能準確執行，且準確度皆在0.99以上，故對CNN架構並無太講究。
