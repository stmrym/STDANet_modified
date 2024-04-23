from models.model.edge_extractor import Edge_extractor_light


if __name__ == '__main__':
    model = Edge_extractor_light(inplanes=1, planes=2, kernel_size=3, stride=1, device='cpu')

    print('-----')
    for name, param in model.named_parameters():
        print(name, param)