from datetime import datetime
import torch


def training_routine(
    cfg, net, trainloader, validationloader, criterion, optimizer, scheduler, device
):
    start_time = datetime.now()
    print(f"Starting training ({datetime.now().strftime('h%H-m%M-s%S')})")
    train_loss_history = []
    validation_loss_history = []
    for epoch in range(cfg.max_epochs):  # loop over the dataset multiple times
        training_running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # update optimizer
            optimizer.step()

            # update LR scheduler
            if scheduler:
                scheduler.step()

            # update loss
            training_running_loss += loss.item()

        # save info
        train_loss_history.append(training_running_loss / len(trainloader))
        training_running_loss = 0.0

        with torch.no_grad():
            validation_running_loss = 0.0
            for i, data in enumerate(validationloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # update loss
                validation_running_loss += loss.item()

            # save stats
            validation_loss_history.append(
                validation_running_loss / len(validationloader)
            )
            validation_running_loss = 0.0

        # print stats
        print(
            f"Epoch [{epoch + 1:04d}\{cfg.max_epochs:0{len(str(cfg.max_epochs))}d}] (elapsed: {(datetime.now()-start_time)}) - Tr loss: {train_loss_history[-1]:.3f} - Val loss: {validation_loss_history[-1]:.3f} - LR: {scheduler.get_last_lr()[0] if scheduler else cfg.base_lr}"
        )
    return train_loss_history, validation_loss_history
